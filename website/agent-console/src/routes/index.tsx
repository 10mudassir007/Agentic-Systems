import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/AppShell";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "AutoFill AI Console — Live Agent Control" },
      {
        name: "description",
        content:
          "Live control panel for an AI agent that autofills web forms via a LangGraph pipeline, with real-time browser streaming and human-in-the-loop input.",
      },
      { property: "og:title", content: "AutoFill AI Console" },
      {
        property: "og:description",
        content:
          "Watch an AI agent read, map, fill, and submit web forms in real time.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
      { name: "twitter:title", content: "AutoFill AI Console" },
      {
        name: "twitter:description",
        content:
          "Watch an AI agent read, map, fill, and submit web forms in real time.",
      },
    ],
  }),
});

function Index() {
  return <AppShell />;
}
