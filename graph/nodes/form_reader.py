from __future__ import annotations

import logging
from typing import Any, Dict, List

from graph.state import GraphState, RunStatus, Field
from services.browser import BrowserManager
from run_registry import registry

logger = logging.getLogger(__name__)


class FormReaderAgent:
    """Navigates to the target URL and extracts every form field via Playwright."""

    def __init__(self) -> None:
        self.browser = BrowserManager()

    async def read_form(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        logger.info("FormReader: loading %s", state["form_url"])

        await registry.publish_status(run_id, stage="form_reader", status="running")

        try:
            page = await self.browser.get_page(run_id)

            await self.browser.start_screencast(run_id)

            await page.goto(state["form_url"], wait_until="networkidle")

            fields = await self._detect_fields(page)
            await self._fix_placeholder_labels(page, fields)
            logger.info("FormReader: found %d fields", len(fields))

            await registry.publish_status(
                run_id, stage="form_reader", status="done", field_count=len(fields)
            )

            return {
                "fields": fields,
                "fields_loaded": True,
                "status": RunStatus.MAPPING_FIELDS,
                "next_agent": "data_mapper",
            }
        except Exception as exc:
            logger.exception("FormReader failed: %s: %s", type(exc).__name__, exc)
            await registry.publish_status(
                run_id,
                stage="form_reader",
                status="error",
                error_message=f"Failed to read form: [{type(exc).__name__}] {exc}",
            )
            return {
                "fields_loaded": False,
                "status": RunStatus.FAILED,
                "error_message": f"Failed to read form: [{type(exc).__name__}] {exc}",
                "next_agent": "__end__",
            }

    async def _detect_fields(self, page) -> List[Field]:
        raw = await page.evaluate("""() => {
            const fields = [];
            const ignoreSet = new Set([
                'fvv', 'partialResponse', 'pageHistory', 'fbzx',
                'submissionTimestamp', 'draftResponse', 'continueUrl',
                'emailReceipt', 'form-shortcode', 'csrfmiddlewaretoken',
                '__requestverificationtoken', '__viewstate',
            ]);

            // Find candidate elements: real inputs and their visible wrappers
            const rawEls = document.querySelectorAll(
                'input, select, textarea, [role="textbox"], [role="combobox"]'
            );
            // Deduplicate: if a role="textbox" contains a real input, skip the wrapper
            const seen = new Set();
            const els = [];
            rawEls.forEach(el => {
                const innerInput = el.querySelector('input, textarea, select');
                if (el.matches('[role="textbox"], [role="combobox"]') && innerInput) {
                    if (!seen.has(innerInput)) {
                        seen.add(innerInput);
                        els.push(innerInput);
                    }
                } else {
                    if (!seen.has(el)) {
                        seen.add(el);
                        els.push(el);
                    }
                }
            });

            els.forEach((el, i) => {
                const f = {
                    field_id: 'field_' + i,
                    field_label: '',
                    field_type: 'short_text',
                    is_required: false,
                    options: null,
                    selector: '',
                    placeholder: null,
                    validation_type: null
                };

                // Skip hidden inputs
                const t = (el.type || '').toLowerCase();
                if (t === 'hidden') return;

                // Skip known internal names (entry.* ARE real form fields)
                const elName = (el.name || '').trim().toLowerCase();
                if (ignoreSet.has(elName)) return;

                // label via for=id
                if (el.id) {
                    const lbl = document.querySelector('label[for="' + el.id + '"]');
                    if (lbl) f.field_label = lbl.textContent.trim();
                }
                // label wrapping
                if (!f.field_label) {
                    const parent = el.closest('label');
                    if (parent) f.field_label = parent.textContent.trim();
                }
                // aria-label
                if (!f.field_label) {
                    f.field_label = el.getAttribute('aria-label') || '';
                }
                // aria-labelledby
                if (!f.field_label) {
                    const labelledBy = el.getAttribute('aria-labelledby');
                    if (labelledBy) {
                        const ref = document.getElementById(labelledBy);
                        if (ref) f.field_label = ref.textContent.trim();
                    }
                }
                // Google Forms: look for preceding span/question text
                if (!f.field_label) {
                    const qs = el.closest('div[role="listitem"]');
                    if (qs) {
                        const labelEl = qs.querySelector(
                            '[role="heading"], span.M7eMe, .M7eMe, div[aria-label], [data-question]'
                        );
                        if (labelEl) {
                            const txt = labelEl.getAttribute('aria-label') || labelEl.textContent;
                            if (txt) f.field_label = txt.trim();
                        }
                        // Broader: look for any non-empty heading-like element
                        if (!f.field_label) {
                            const headings = qs.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"], .freebirdFormviewerViewItemsItemItemTitle');
                            for (const h of headings) {
                                const t = h.textContent.trim();
                                if (t) { f.field_label = t; break; }
                            }
                        }
                        // Aggressive: scan all text nodes for the first substantial sentence
                        if (!f.field_label) {
                            const placeholderWords = ['your answer', 'your response', 'type here', 'write here', 'enter text'];
                            const walker = document.createTreeWalker(qs, NodeFilter.SHOW_TEXT, null, false);
                            const parts = [];
                            while (walker.nextNode()) {
                                const t = walker.currentNode.textContent.trim();
                                if (!t || t.length < 3) continue;
                                if (t === el.placeholder) continue;
                                const lower = t.toLowerCase();
                                if (placeholderWords.some(w => lower === w)) continue;
                                parts.push(t);
                            }
                            if (parts.length) {
                                parts.sort((a, b) => b.length - a.length);
                                f.field_label = parts[0];
                            }
                        }
                    }
                }
                // Final fallback: scan parent chain for any heading or label
                if (!f.field_label) {
                    let cur = el.parentElement;
                    let maxDepth = 5;
                    while (cur && maxDepth-- > 0) {
                        for (const child of cur.children) {
                            const tag = child.tagName;
                            if (['H1','H2','H3','H4','H5','H6','LABEL'].includes(tag)) {
                                const txt = child.textContent.trim();
                                if (txt) { f.field_label = txt; break; }
                            }
                            if (child.getAttribute('role') === 'heading') {
                                const txt = child.textContent.trim();
                                if (txt) { f.field_label = txt; break; }
                            }
                        }
                        if (f.field_label) break;
                        cur = cur.parentElement;
                    }
                }
                // placeholder / name / fallback
                if (!f.field_label) {
                    f.field_label = el.placeholder || el.name || ('Field ' + (i + 1));
                }

                // Skip if label is still empty or looks like a technical name
                const label = f.field_label.trim();
                if (!label || /^[a-z0-9_.-]+$/.test(label) || label.startsWith('field_') || label.startsWith('entry.')) return;

                // type
                if (el.tagName === 'SELECT') {
                    f.field_type = 'dropdown';
                    f.options = Array.from(el.options).map(o => o.text.trim());
                } else if (t === 'checkbox')     f.field_type = 'checkbox';
                else if (t === 'radio')          f.field_type = 'radio';
                else if (t === 'date')           f.field_type = 'date';
                else if (el.tagName === 'TEXTAREA') f.field_type = 'long_text';
                else if (t === 'email')          { f.field_type = 'email'; f.validation_type = 'email'; }
                else if (t === 'tel')            { f.field_type = 'phone'; f.validation_type = 'phone'; }
                else if (t === 'number')         { f.field_type = 'number'; f.validation_type = 'number'; }

                f.is_required = el.required || el.hasAttribute('required');

                // best-effort selector
                if (el.id)                          f.selector = '#' + el.id;
                else if (el.name)                   f.selector = '[name="' + el.name + '"]';
                else {
                    // Google Forms: use listitem index
                    const qs = el.closest('div[role="listitem"], div[role="group"]');
                    if (qs) {
                        const parent = qs.parentElement;
                        if (parent) {
                            const siblings = parent.querySelectorAll(':scope > div[role="listitem"], :scope > div[role="group"]');
                            const idx = Array.from(siblings).indexOf(qs);
                            if (idx >= 0) {
                                const tag = el.tagName.toLowerCase();
                                f.selector = `[role="listitem"]:nth-of-type(${idx + 1}) ${tag}`;
                            }
                        }
                    }
                    if (!f.selector) {
                        f.selector = '[data-field-id="' + f.field_id + '"]';
                    }
                }

                f.placeholder = el.placeholder || null;
                fields.push(f);
            });
            return fields;
        }""")

        return [Field(**f) for f in raw]

    async def _fix_placeholder_labels(self, page, fields: List[Field]) -> None:
        placeholders = {"your answer", "your response", "type here", "write here"}
        indices = [i for i, f in enumerate(fields) if f["field_label"].lower().strip() in placeholders]
        if not indices:
            return
        corrections = await page.evaluate("""(idxs) => {
            const rawEls = document.querySelectorAll(
                'input, select, textarea, [role="textbox"], [role="combobox"]'
            );
            const seen = new Set();
            const els = [];
            rawEls.forEach(el => {
                const innerInput = el.querySelector('input, textarea, select');
                if (el.matches('[role="textbox"], [role="combobox"]') && innerInput) {
                    if (!seen.has(innerInput)) { seen.add(innerInput); els.push(innerInput); }
                } else {
                    if (!seen.has(el)) { seen.add(el); els.push(el); }
                }
            });
            const ignoreSet = new Set(['fvv','partialResponse','pageHistory','fbzx','submissionTimestamp','draftResponse','continueUrl','emailReceipt','form-shortcode','csrfmiddlewaretoken','__requestverificationtoken','__viewstate']);
            const filtered = els.filter(el => {
                const t = (el.type || '').toLowerCase();
                const elName = (el.name || '').trim().toLowerCase();
                return t !== 'hidden' && !ignoreSet.has(elName);
            });
            const skipWords = ['your answer', 'your response', 'type here', 'write here'];
            const results = [];
            for (const idx of idxs) {
                const el = filtered[idx];
                if (!el) { results.push(null); continue; }
                const listitem = el.closest('div[role="listitem"]');
                if (!listitem) {
                    let cur = el.parentElement;
                    let depth = 0;
                    let found = null;
                    while (cur && cur.tagName !== 'BODY' && depth++ < 10) {
                        for (const child of cur.children) {
                            if (['H1','H2','H3','H4','H5','H6','LABEL'].includes(child.tagName)) {
                                const t = child.textContent.trim();
                                if (t && t.length > 3) { found = t; break; }
                            }
                        }
                        if (found) break;
                        cur = cur.parentElement;
                    }
                    results.push(found);
                } else {
                    const walker = document.createTreeWalker(listitem, NodeFilter.SHOW_TEXT, null, false);
                    const parts = [];
                    while (walker.nextNode()) {
                        const t = walker.currentNode.textContent.trim();
                        if (!t || t.length < 5) continue;
                        if (skipWords.includes(t.toLowerCase())) continue;
                        parts.push(t);
                    }
                    if (parts.length) {
                        parts.sort((a, b) => b.length - a.length);
                        results.push(parts[0]);
                    } else {
                        results.push(null);
                    }
                }
            }
            return results;
        }""", indices)
        for idx, label in zip(indices, corrections):
            if label:
                fields[idx]["field_label"] = label
