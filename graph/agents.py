# backend/graph/nodes/supervisor.py
from typing import Dict, Any, List, Optional
from langgraph.types import interrupt, Command
import logging
from datetime import datetime

from ..state import GraphState, RunStatus, FieldStatus, FieldResult, Field
from ...services.llm import LLMClient
from ...config import config

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """Hub-and-spoke orchestrator. All nodes route back through here."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.max_retries = config.MAX_RETRIES_PER_FIELD
    
    def supervise(self, state: GraphState) -> Dict[str, Any]:
        """Main supervisor decision node."""
        logger.info(f"Supervisor: Run {state['run_id']} - Status: {state['status']}")
        
        updates = {}
        state_copy = state.copy()
        
        # Initialize timestamps
        if not state_copy.get("started_at"):
            updates["started_at"] = datetime.now().isoformat()
        
        # Check for human interrupt
        if state_copy.get("awaiting_human"):
            updates["status"] = RunStatus.WAITING_FOR_HUMAN
            updates["next_agent"] = "waiting"
            return updates
        
        # Check if form needs to be loaded
        if not state_copy.get("fields_loaded"):
            updates["status"] = RunStatus.READING_FORM
            updates["next_agent"] = "form_reader"
            return updates
        
        # Check if fields need mapping
        if state_copy.get("fields_loaded") and not self._has_mapping_results(state_copy):
            updates["status"] = RunStatus.MAPPING_FIELDS
            updates["next_agent"] = "data_mapper"
            return updates
        
        # Handle unresolved fields
        unresolved = state_copy.get("unresolved_fields", {})
        if unresolved:
            return self._handle_unresolved_fields(state_copy, unresolved)
        
        # Handle filler results
        filler_results = state_copy.get("filler_results", {})
        if filler_results:
            return self._handle_filler_results(state_copy, filler_results)
        
        # Check if ready to fill
        resolved = state_copy.get("resolved_fields", {})
        if resolved and not filler_results:
            updates["status"] = RunStatus.FILLING_FORM
            updates["next_agent"] = "filler"
            return updates
        
        # Check completion
        if self._is_complete(state_copy):
            updates["status"] = RunStatus.COMPLETED
            updates["completion_message"] = "Form filled successfully!"
            updates["completed_at"] = datetime.now().isoformat()
            updates["next_agent"] = "end"
            return updates
        
        # Default end
        updates["status"] = RunStatus.COMPLETED
        updates["completion_message"] = "No more work to do"
        updates["completed_at"] = datetime.now().isoformat()
        updates["next_agent"] = "end"
        return updates
    
    def _has_mapping_results(self, state: GraphState) -> bool:
        return bool(state.get("resolved_fields")) or bool(state.get("unresolved_fields"))
    
    def _handle_unresolved_fields(self, state: GraphState, 
                                  unresolved: Dict[str, FieldResult]) -> Dict[str, Any]:
        """Handle fields that Data Mapper couldn't resolve."""
        updates = {}
        field_ids = list(unresolved.keys())
        retry_counts = state.get("field_retry_counts", {})
        
        need_human = False
        retry_fields = {}
        
        for field_id, result in unresolved.items():
            retry_count = retry_counts.get(field_id, 0)
            
            if retry_count >= self.max_retries:
                need_human = True
                break
            else:
                retry_counts[field_id] = retry_count + 1
                retry_fields[field_id] = result
        
        if need_human:
            return self._request_human_input(state, unresolved)
        
        updates["field_retry_counts"] = retry_counts
        updates["unresolved_fields"] = retry_fields
        updates["status"] = RunStatus.MAPPING_FIELDS
        updates["next_agent"] = "data_mapper"
        updates["retry_count"] = state.get("retry_count", 0) + 1
        
        logger.info(f"Retrying {len(retry_fields)} unresolved fields")
        return updates
    
    def _request_human_input(self, state: GraphState, 
                            unresolved: Dict[str, FieldResult]) -> Dict[str, Any]:
        """Pause for human input using LangGraph interrupt."""
        fields = state.get("fields", [])
        field_map = {f["field_id"]: f for f in fields}
        
        questions = []
        for field_id, result in unresolved.items():
            field = field_map.get(field_id, {})
            question = {
                "field_id": field_id,
                "question": f"Please provide a value for: {field.get('field_label', 'Unknown field')}",
                "field_type": field.get("field_type", "short_text"),
                "is_required": field.get("is_required", True),
                "options": field.get("options", []),
                "current_value": result.get("value"),
                "confidence": result.get("confidence", 0),
            }
            questions.append(question)
        
        interrupt_data = {
            "run_id": state["run_id"],
            "questions": questions,
            "prompt": "Human input required to complete form fields"
        }
        
        try:
            human_response = interrupt(interrupt_data)
            
            if isinstance(human_response, dict):
                answers = human_response.get("answers", {})
                return self._process_human_answers(state, answers)
            else:
                logger.error(f"Invalid human response format: {human_response}")
                return {
                    "status": RunStatus.FAILED,
                    "error_message": "Invalid human input format received",
                    "next_agent": "end"
                }
                
        except Exception as e:
            logger.error(f"Human interrupt failed: {e}")
            return {
                "status": RunStatus.FAILED,
                "error_message": f"Failed to get human input: {str(e)}",
                "next_agent": "end"
            }
    
    def _process_human_answers(self, state: GraphState, 
                              answers: Dict[str, str]) -> Dict[str, Any]:
        """Process human-provided answers and route back to Data Mapper."""
        updates = {}
        
        human_answers = state.get("human_answers", {})
        human_answers.update(answers)
        updates["human_answers"] = human_answers
        
        unresolved = state.get("unresolved_fields", {})
        resolved = state.get("resolved_fields", {})
        
        for field_id, answer in answers.items():
            if field_id in unresolved:
                result = unresolved[field_id]
                result["value"] = answer
                result["status"] = FieldStatus.RESOLVED
                result["confidence"] = 1.0
                resolved[field_id] = result
                del unresolved[field_id]
        
        updates["unresolved_fields"] = unresolved
        updates["resolved_fields"] = resolved
        updates["awaiting_human"] = False
        updates["status"] = RunStatus.MAPPING_FIELDS
        updates["next_agent"] = "data_mapper"
        
        logger.info(f"Processed human answers for {len(answers)} fields")
        return updates
    
    def _handle_filler_results(self, state: GraphState, 
                              filler_results: Dict[str, FieldResult]) -> Dict[str, Any]:
        """Handle results from Filler Agent."""
        updates = {}
        
        errors = []
        for field_id, result in filler_results.items():
            if result.get("status") == FieldStatus.FILL_ERROR:
                errors.append((field_id, result))
        
        if not errors:
            if self._is_complete(state):
                updates["status"] = RunStatus.COMPLETED
                updates["completion_message"] = "Form filled successfully!"
                updates["completed_at"] = datetime.now().isoformat()
                updates["next_agent"] = "end"
            else:
                updates["status"] = RunStatus.COMPLETED
                updates["completion_message"] = "All fields processed"
                updates["completed_at"] = datetime.now().isoformat()
                updates["next_agent"] = "end"
            return updates
        
        # Handle errors
        field_retry_counts = state.get("field_retry_counts", {})
        need_human = False
        retry_fields = {}
        
        for field_id, result in errors:
            retry_count = field_retry_counts.get(field_id, 0)
            
            if retry_count >= self.max_retries:
                need_human = True
                break
            else:
                field_retry_counts[field_id] = retry_count + 1
                retry_fields[field_id] = result
        
        if need_human:
            return self._request_human_input(state, {
                field_id: result 
                for field_id, result in errors 
                if field_id in field_retry_counts and field_retry_counts[field_id] >= self.max_retries
            })
        
        updates["filler_results"] = {}
        updates["field_retry_counts"] = field_retry_counts
        updates["status"] = RunStatus.FILLING_FORM
        updates["next_agent"] = "filler"
        updates["retry_count"] = state.get("retry_count", 0) + 1
        
        logger.info(f"Retrying {len(retry_fields)} filler errors")
        return updates
    
    def _is_complete(self, state: GraphState) -> bool:
        resolved = state.get("resolved_fields", {})
        filler_results = state.get("filler_results", {})
        
        if not resolved:
            return False
        
        all_filled = all(
            field_id in filler_results 
            for field_id in resolved.keys()
        )
        
        has_errors = any(
            result.get("status") == FieldStatus.FILL_ERROR 
            for result in filler_results.values()
        )
        
        return all_filled and not has_errors
# backend/graph/nodes/form_reader.py
import asyncio
import logging
from typing import Dict, Any, List
from playwright.async_api import async_playwright, Page, Browser

from ..state import GraphState, RunStatus, Field
from ...services.browser import BrowserManager

logger = logging.getLogger(__name__)

class FormReaderAgent:
    """Reads form fields from the target URL using Playwright."""
    
    def __init__(self):
        self.browser_manager = BrowserManager()
    
    async def read_form(self, state: GraphState) -> Dict[str, Any]:
        """Extract all form fields from the target URL."""
        logger.info(f"FormReader: Reading form from {state['form_url']}")
        
        try:
            page = await self.browser_manager.get_page(state["run_id"])
            
            # Navigate to the form
            await page.goto(state["form_url"], wait_until="networkidle")
            
            # Detect all form fields
            fields = await self._detect_fields(page)
            
            logger.info(f"FormReader: Found {len(fields)} fields")
            
            return {
                "fields": fields,
                "fields_loaded": True,
                "status": RunStatus.MAPPING_FIELDS,
                "next_agent": "data_mapper"
            }
            
        except Exception as e:
            logger.error(f"FormReader failed: {e}")
            return {
                "fields_loaded": False,
                "status": RunStatus.FAILED,
                "error_message": f"Failed to read form: {str(e)}",
                "next_agent": "end"
            }
    
    async def _detect_fields(self, page: Page) -> List[Field]:
        """Detect all form fields using Playwright selectors."""
        fields = []
        
        # Use JavaScript to extract all form fields
        field_data = await page.evaluate("""
            () => {
                const fields = [];
                const inputs = document.querySelectorAll('input, select, textarea, [role="textbox"]');
                
                inputs.forEach((el, index) => {
                    const field = {
                        field_id: `field_${index}`,
                        field_label: '',
                        field_type: 'short_text',
                        is_required: false,
                        options: null,
                        selector: '',
                        placeholder: '',
                        validation_type: null
                    };
                    
                    // Get label
                    if (el.id) {
                        const label = document.querySelector(`label[for="${el.id}"]`);
                        if (label) {
                            field.field_label = label.textContent.trim();
                        }
                    }
                    
                    if (!field.field_label) {
                        const parentLabel = el.closest('label');
                        if (parentLabel) {
                            field.field_label = parentLabel.textContent.trim();
                        }
                    }
                    
                    if (!field.field_label) {
                        field.field_label = el.placeholder || el.name || `Field ${index + 1}`;
                    }
                    
                    // Get field type
                    if (el.tagName === 'SELECT') {
                        field.field_type = 'dropdown';
                        field.options = Array.from(el.options).map(opt => opt.textContent);
                    } else if (el.type === 'checkbox') {
                        field.field_type = 'checkbox';
                    } else if (el.type === 'radio') {
                        field.field_type = 'radio';
                    } else if (el.type === 'date') {
                        field.field_type = 'date';
                    } else if (el.tagName === 'TEXTAREA') {
                        field.field_type = 'long_text';
                    } else if (el.type === 'email') {
                        field.field_type = 'email';
                        field.validation_type = 'email';
                    } else if (el.type === 'number') {
                        field.field_type = 'number';
                        field.validation_type = 'number';
                    }
                    
                    // Check if required
                    field.is_required = el.required || el.hasAttribute('required');
                    
                    // Create selector
                    if (el.id) {
                        field.selector = `#${el.id}`;
                    } else if (el.name) {
                        field.selector = `[name="${el.name}"]`;
                    } else {
                        field.selector = `[data-field-index="${index}"]`;
                    }
                    
                    field.placeholder = el.placeholder || null;
                    
                    fields.push(field);
                });
                
                return fields;
            }
        """)
        
        # Convert to Field type
        for field in field_data:
            fields.append(Field(
                field_id=field["field_id"],
                field_label=field["field_label"],
                field_type=field["field_type"],
                is_required=field["is_required"],
                options=field.get("options"),
                selector=field["selector"],
                placeholder=field.get("placeholder"),
                validation_type=field.get("validation_type")
            ))
        
        return fields
# backend/graph/nodes/data_mapper.py
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

from ..state import GraphState, RunStatus, FieldStatus, FieldResult
from ...services.vector_store import VectorStore
from ...services.llm import LLMClient
from ...config import config

logger = logging.getLogger(__name__)

class DataMapperAgent:
    """Maps form fields to bio data using LLM reasoning."""
    
    def __init__(self):
        self.llm = LLMClient()
        self.vector_store = VectorStore()
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
    
    async def map_fields(self, state: GraphState) -> Dict[str, Any]:
        """Map form fields to bio data values."""
        logger.info(f"DataMapper: Mapping fields for run {state['run_id']}")
        
        fields = state.get("fields", [])
        if not fields:
            return {
                "status": RunStatus.FAILED,
                "error_message": "No fields to map",
                "next_agent": "end"
            }
        
        # Get bio data for this run (loaded from vector store)
        bio_data = await self._get_bio_data(state["run_id"])
        
        # Process fields in batches
        batch_size = config.BATCH_SIZE
        resolved_fields = state.get("resolved_fields", {})
        unresolved_fields = state.get("unresolved_fields", {})
        
        # Only process fields that haven't been resolved yet
        to_process = [
            f for f in fields 
            if f["field_id"] not in resolved_fields 
            and f["field_id"] not in unresolved_fields
        ]
        
        if not to_process:
            # All fields already processed
            return {
                "status": RunStatus.FILLING_FORM,
                "next_agent": "filler"
            }
        
        # Process in batches
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i+batch_size]
            batch_results = await self._map_batch(batch, bio_data)
            
            for result in batch_results:
                if result["status"] == FieldStatus.RESOLVED:
                    resolved_fields[result["field_id"]] = result
                else:
                    unresolved_fields[result["field_id"]] = result
        
        # Check if any fields are unresolved
        next_agent = "filler" if not unresolved_fields else "supervisor"
        
        return {
            "resolved_fields": resolved_fields,
            "unresolved_fields": unresolved_fields,
            "status": RunStatus.FILLING_FORM if not unresolved_fields else RunStatus.MAPPING_FIELDS,
            "next_agent": next_agent
        }
    
    async def _map_batch(self, fields: List[Dict], bio_data: Dict) -> List[FieldResult]:
        """Map a batch of fields using LLM."""
        results = []
        
        for field in fields:
            result = await self._map_single_field(field, bio_data)
            results.append(result)
        
        return results
    
    async def _map_single_field(self, field: Dict, bio_data: Dict) -> FieldResult:
        """Map a single field using LLM reasoning."""
        
        # Check learned Q&A pairs first
        learned_answer = await self._check_learned_pairs(field["field_label"])
        if learned_answer and learned_answer["confidence"] >= self.confidence_threshold:
            return FieldResult(
                field_id=field["field_id"],
                status=FieldStatus.RESOLVED,
                value=learned_answer["value"],
                confidence=learned_answer["confidence"],
                timestamp=datetime.now().isoformat()
            )
        
        # Use LLM to map field to bio data
        prompt = self._build_mapping_prompt(field, bio_data)
        
        try:
            response = await self.llm.async_complete(prompt)
            mapped_value = self._parse_llm_response(response, field)
            
            if mapped_value and mapped_value["confidence"] >= self.confidence_threshold:
                return FieldResult(
                    field_id=field["field_id"],
                    status=FieldStatus.RESOLVED,
                    value=mapped_value["value"],
                    confidence=mapped_value["confidence"],
                    timestamp=datetime.now().isoformat()
                )
            else:
                return FieldResult(
                    field_id=field["field_id"],
                    status=FieldStatus.UNRESOLVED,
                    confidence=mapped_value["confidence"] if mapped_value else 0.0,
                    human_question=f"Could you provide a value for: {field['field_label']}",
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"LLM mapping failed for {field['field_label']}: {e}")
            return FieldResult(
                field_id=field["field_id"],
                status=FieldStatus.UNRESOLVED,
                confidence=0.0,
                human_question=f"Error mapping: {field['field_label']}",
                timestamp=datetime.now().isoformat()
            )
    
    def _build_mapping_prompt(self, field: Dict, bio_data: Dict) -> str:
        """Build prompt for LLM mapping."""
        return f"""
        You are a data mapping assistant. Map the following form field to the appropriate value from the user's bio data.
        
        Form Field:
        Label: {field['field_label']}
        Type: {field['field_type']}
        Required: {field['is_required']}
        Options: {field.get('options', [])}
        
        Bio Data:
        {bio_data}
        
        Return a JSON object with:
        1. "value": the mapped value (or null if not found)
        2. "confidence": a number between 0 and 1 indicating how confident you are
        3. "explanation": brief explanation of your reasoning
        
        Only return the JSON object.
        """
    
    def _parse_llm_response(self, response: str, field: Dict) -> Optional[Dict]:
        """Parse LLM response to extract value and confidence."""
        import json
        
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return {
                    "value": data.get("value"),
                    "confidence": data.get("confidence", 0.0),
                    "explanation": data.get("explanation", "")
                }
        except:
            pass
        
        return None
    
    async def _check_learned_pairs(self, field_label: str) -> Optional[Dict]:
        """Check vector store for learned Q&A pairs."""
        try:
            results = await self.vector_store.similarity_search(
                field_label,
                top_k=1,
                threshold=self.confidence_threshold
            )
            
            if results:
                return {
                    "value": results[0]["answer"],
                    "confidence": results[0]["similarity"]
                }
        except:
            pass
        
        return None
    
    async def _get_bio_data(self, run_id: str) -> Dict:
        """Retrieve bio data for this run."""
        # In production, this would load from database
        # For demo, return sample data
        return {
            "full_name": "John Smith",
            "email": "john.smith@email.com",
            "phone": "+1-555-123-4567",
            "occupation": "Software Engineer",
            "job_title": "Senior Software Engineer",
            "experience_years": 8,
            "education": "Bachelor's Degree",
            "tech_stack": ["Python", "JavaScript/TypeScript", "Go"],
            "availability_start": "2026-01-15",
            "preferred_hours": "Flexible"
        }

# backend/graph/nodes/filler.py
import logging
import asyncio
from typing import Dict, Any, List, Optional

from ..state import GraphState, RunStatus, FieldStatus, FieldResult
from ...services.browser import BrowserManager

logger = logging.getLogger(__name__)

class FillerAgent:
    """Fills form fields using Playwright."""
    
    def __init__(self):
        self.browser_manager = BrowserManager()
    
    async def fill_fields(self, state: GraphState) -> Dict[str, Any]:
        """Fill all resolved fields."""
        logger.info(f"Filler: Filling fields for run {state['run_id']}")
        
        resolved_fields = state.get("resolved_fields", {})
        if not resolved_fields:
            return {
                "status": RunStatus.FAILED,
                "error_message": "No resolved fields to fill",
                "next_agent": "end"
            }
        
        page = await self.browser_manager.get_page(state["run_id"])
        filler_results = {}
        errors = []
        
        for field_id, result in resolved_fields.items():
            # Skip if already filled
            if field_id in state.get("filler_results", {}):
                continue
            
            try:
                success = await self._fill_single_field(
                    page,
                    field_id,
                    result,
                    state.get("fields", [])
                )
                
                if success:
                    filler_results[field_id] = FieldResult(
                        field_id=field_id,
                        status=FieldStatus.FILLED,
                        value=result["value"],
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    filler_results[field_id] = FieldResult(
                        field_id=field_id,
                        status=FieldStatus.FILL_ERROR,
                        value=result["value"],
                        error_message="Failed to fill field",
                        timestamp=datetime.now().isoformat()
                    )
                    errors.append(field_id)
                    
            except Exception as e:
                logger.error(f"Failed to fill field {field_id}: {e}")
                filler_results[field_id] = FieldResult(
                    field_id=field_id,
                    status=FieldStatus.FILL_ERROR,
                    value=result["value"],
                    error_message=str(e),
                    timestamp=datetime.now().isoformat()
                )
                errors.append(field_id)
        
        # Check if any errors occurred
        if errors:
            return {
                "filler_results": filler_results,
                "filler_errors": errors,
                "status": RunStatus.FILLING_FORM,
                "next_agent": "supervisor"  # Go back to supervisor for retry logic
            }
        
        return {
            "filler_results": filler_results,
            "filler_errors": [],
            "status": RunStatus.COMPLETED,
            "next_agent": "end",
            "completion_message": "Form filled successfully!"
        }
    
    async def _fill_single_field(self, page, field_id: str, 
                                 result: FieldResult, fields: List[Dict]) -> bool:
        """Fill a single field with Playwright."""
        # Find the field
        field = next((f for f in fields if f["field_id"] == field_id), None)
        if not field:
            logger.warning(f"Field {field_id} not found in field list")
            return False
        
        selector = field.get("selector")
        if not selector:
            logger.warning(f"No selector for field {field_id}")
            return False
        
        value = result["value"]
        if not value:
            logger.warning(f"No value for field {field_id}")
            return False
        
        field_type = field.get("field_type", "short_text")
        
        try:
            # Wait for element
            element = await page.wait_for_selector(selector, timeout=5000)
            if not element:
                return False
            
            # Fill based on field type
            if field_type in ["dropdown", "select"]:
                await element.select_option(value)
            elif field_type in ["checkbox", "radio"]:
                if value.lower() in ["true", "yes", "on", "1"]:
                    await element.check()
                else:
                    await element.uncheck()
            elif field_type == "date":
                await element.fill(value)
            else:
                # Clear and fill
                await element.click()
                await element.clear()
                await element.fill(value)
            
            # Trigger change event
            await page.evaluate("(el) => el.dispatchEvent(new Event('change', { bubbles: true }))", element)
            
            return True
            
        except Exception as e:
            logger.error(f"Error filling field {field_id}: {e}")
            return False
# backend/services/browser.py
import asyncio
from typing import Dict, Optional
from playwright.async_api import async_playwright, Browser, Page, Playwright
import logging

logger = logging.getLogger(__name__)

class BrowserManager:
    """Manages Playwright browser sessions."""
    
    _instance = None
    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _pages: Dict[str, Page] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Initialize Playwright browser."""
        if not self._playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=False,  # Headed mode for visibility
                args=['--start-maximized']
            )
            logger.info("Browser initialized")
    
    async def get_page(self, run_id: str) -> Page:
        """Get or create a page for a run."""
        await self.initialize()
        
        if run_id not in self._pages:
            context = await self._browser.new_context(
                viewport={'width': 1280, 'height': 720}
            )
            page = await context.new_page()
            self._pages[run_id] = page
            logger.info(f"Created new page for run {run_id}")
        
        return self._pages[run_id]
    
    async def close_page(self, run_id: str):
        """Close a page for a run."""
        if run_id in self._pages:
            await self._pages[run_id].close()
            del self._pages[run_id]
            logger.info(f"Closed page for run {run_id}")
    
    async def close_all(self):
        """Close all pages and browser."""
        for run_id, page in self._pages.items():
            await page.close()
        self._pages.clear()
        
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        logger.info("Closed all browser instances")