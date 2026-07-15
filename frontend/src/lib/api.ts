import axios from "axios";
import type {
  AssistantChatResponse,
  AssistantMessage,
  ChatMessage,
  ChatResponse,
  HistoryItem,
  ResearchSession,
  ResearchStartResponse,
  ResearchStatusResponse,
} from "@/types";

const api = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

export async function startResearch(topic: string): Promise<ResearchStartResponse> {
  const { data } = await api.post<ResearchStartResponse>("/research", { topic });
  return data;
}

export async function getResearchStatus(sessionId: string): Promise<ResearchStatusResponse> {
  const { data } = await api.get<ResearchStatusResponse>(`/research/${sessionId}/status`);
  return data;
}

export async function getResearchReport(sessionId: string): Promise<ResearchSession> {
  const { data } = await api.get<ResearchSession>(`/research/${sessionId}`);
  return data;
}

export async function getHistory(): Promise<HistoryItem[]> {
  const { data } = await api.get<HistoryItem[]>("/history");
  return data;
}

export function exportUrl(sessionId: string, format: "markdown" | "text" | "pdf" | "docx"): string {
  return `/api/research/${sessionId}/export/${format}`;
}

export async function sendChatMessage(sessionId: string, message: string): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>(`/research/${sessionId}/chat`, { message });
  return data;
}

export async function getChatHistory(sessionId: string): Promise<ChatMessage[]> {
  const { data } = await api.get<ChatMessage[]>(`/research/${sessionId}/chat`);
  return data;
}

export async function sendAssistantMessage(message: string): Promise<AssistantChatResponse> {
  const { data } = await api.post<AssistantChatResponse>("/assistant/chat", { message });
  return data;
}

export async function getAssistantHistory(): Promise<AssistantMessage[]> {
  const { data } = await api.get<AssistantMessage[]>("/assistant/chat");
  return data;
}

export async function clearAssistantHistory(): Promise<void> {
  await api.delete("/assistant/chat");
}

export default api;
