'use client';

import { useState, useEffect, useRef, useCallback, FormEvent } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { X, Send, Bot, Minimize2, Maximize2, Lightbulb, AlertTriangle, FolderOpen, CheckCircle, Loader, WifiOff } from 'lucide-react';
import type { LlmInsight } from './LlmEventBridge';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'user' | 'assistant';
  tag?: LlmInsight['tag'];
  title?: string;
  body: string;
  ts: string;
}

// (Toast popup system removed — insights appear in chat history only)

// ---------------------------------------------------------------------------
// Chat message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`copilot-bubble ${isUser ? 'copilot-bubble--user' : 'copilot-bubble--bot'}`}>
      {!isUser && (
        <div className="copilot-bubble__meta">
          {msg.tag === 'alert'
            ? <AlertTriangle size={11} className="copilot-bubble__tag-icon copilot-bubble__tag-icon--alert" />
            : msg.tag === 'advice'
            ? <Lightbulb size={11} className="copilot-bubble__tag-icon copilot-bubble__tag-icon--advice" />
            : <Bot size={11} className="copilot-bubble__tag-icon" />
          }
          <span className="copilot-bubble__title">{msg.title ?? 'Co-Pilot'}</span>
          <span className="copilot-bubble__ts">{msg.ts}</span>
        </div>
      )}
      <p className="copilot-bubble__body">{msg.body}</p>
      {isUser && <span className="copilot-bubble__ts copilot-bubble__ts--user">{msg.ts}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Co-Pilot sidebar
// ---------------------------------------------------------------------------

interface CoPilotProps {
  /** New non-chat insight pushed from Rust (advice / alert), to auto-add to history */
  latestInsight: LlmInsight | null;
  /** Currently selected training env from the main page (e.g. 'conda:nocode_train') */
  selectedEnv: string;
}

// Parse status string into display-friendly shape
function parseStatus(raw: string): { label: string; color: 'green' | 'orange' | 'red' | 'gray' } {
  if (raw.startsWith('ready:'))  return { label: raw.slice(6), color: 'green' };
  if (raw === 'loading')          return { label: 'Loading…',   color: 'orange' };
  if (raw.startsWith('error:'))   return { label: 'Error',      color: 'red' };
  return { label: 'No model',     color: 'gray' };
}

function StatusBadge({ raw }: { raw: string }) {
  const { label, color } = parseStatus(raw);
  const icon =
    color === 'green'  ? <CheckCircle size={10} /> :
    color === 'orange' ? <Loader size={10} className="copilot-spin" /> :
    color === 'red'    ? <WifiOff size={10} /> :
                         <WifiOff size={10} />;

  const maxLen = 18;
  const display = label.length > maxLen ? `…${label.slice(-(maxLen - 1))}` : label;

  return (
    <span className={`copilot-status-badge copilot-status-badge--${color}`} title={label}>
      {icon}
      <span className="copilot-status-badge__text">{display}</span>
    </span>
  );
}


export default function CoPilotSidebar({ latestInsight, selectedEnv }: CoPilotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [llmStatus, setLlmStatus] = useState('not_loaded');
  const [loadingModel, setLoadingModel] = useState(false);

  // Keep latest selectedEnv in a ref so callbacks always see the current value
  const selectedEnvRef = useRef(selectedEnv);
  useEffect(() => { selectedEnvRef.current = selectedEnv; }, [selectedEnv]);

  // Helper: extract bare conda env name from 'conda:name' or return ''
  const bareEnv = (env: string) => env.startsWith('conda:') ? env.slice(6) : '';

  // Auto-respawn the LLM daemon whenever a conda GPU env is selected
  useEffect(() => {
    const env = bareEnv(selectedEnv);
    if (!env) return; // system Python — don't auto-respawn; user can click load
    invoke('set_llm_conda_env', { condaEnv: env }).catch(console.error);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedEnv]);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      tag: 'advice',
      title: 'EPOQ Co-Pilot',
      body: 'Hi! I\'m your AI training assistant. I\'ll proactively analyze your training metrics and alert you to any issues. You can also ask me anything about your run.',
      ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
    },
  ]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Poll LLM status every 3 seconds
  useEffect(() => {
    const poll = async () => {
      try {
        const s = await invoke<string>('get_llm_status');
        setLlmStatus(s);
      } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  // Pick .gguf file and load it
  const handleLoadModel = useCallback(async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: 'GGUF Model', extensions: ['gguf'] }],
        title: 'Select GGUF Model File',
      });
      if (!selected || typeof selected !== 'string') return;
      setLoadingModel(true);
      const env = bareEnv(selectedEnvRef.current);
      await invoke('load_llm_model', { path: selected, condaEnv: env || null });
      setLlmStatus('loading');
    } catch (err) {
      console.error('Failed to load model:', err);
    } finally {
      setLoadingModel(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When daemon becomes ready, update local status immediately from insight
  useEffect(() => {
    if (latestInsight?.title?.startsWith('LLM Ready')) {
      invoke<string>('get_llm_status').then(setLlmStatus).catch(() => {});
    }
  }, [latestInsight]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // When a new non-chat insight arrives from the LLM, append to chat history (no popup)
  useEffect(() => {
    if (!latestInsight || latestInsight.tag === 'chat') return;
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        tag: latestInsight.tag,
        title: latestInsight.title,
        body: latestInsight.body,
        ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
      },
    ]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestInsight]);

  const handleSend = async (e?: FormEvent) => {
    e?.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setInput('');
    setSending(true);
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        body: text,
        ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
      },
    ]);

    try {
      await invoke('send_chat_message', { message: text });
      // The response will arrive via the `llm_insight` event → latestInsight prop
      // We handle chat responses in page.tsx and pass them back via latestInsight,
      // but for chat the tag="chat" so we handle it here separately:
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          tag: 'alert',
          title: 'Error',
          body: `Failed to send: ${err}`,
          ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  // Handle incoming chat responses (tag === 'chat') from parent
  // We expose a method via a hidden div id so page.tsx can push chat responses
  // Actually, we'll handle this via latestInsight for chat too:
  useEffect(() => {
    if (!latestInsight || latestInsight.tag !== 'chat') return;
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        tag: 'chat',
        title: latestInsight.title,
        body: latestInsight.body,
        ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
      },
    ]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestInsight]);

  return (
    <>

      {/* Floating trigger button */}
      <button
        id="copilot-trigger"
        onClick={() => { setIsOpen(true); setIsMinimized(false); }}
        className="copilot-fab"
        title="Open AI Co-Pilot"
        aria-label="Open AI Co-Pilot"
      >
        <Bot size={22} />
        {/* Pulse only when model is ready */}
        {llmStatus.startsWith('ready:') && <span className="copilot-fab__pulse" />}
      </button>

      {/* Sidebar drawer */}
      {isOpen && (
        <div className={`copilot-drawer ${isMinimized ? 'copilot-drawer--minimized' : ''}`}>
          {/* Header */}
          <div className="copilot-drawer__header">
            <div className="copilot-drawer__header-left">
              <Bot size={16} className="copilot-drawer__header-icon" />
              <span className="copilot-drawer__title">EPOQ Co-Pilot</span>
              <span className="copilot-drawer__badge">AI</span>
              {/* LLM status badge */}
              <StatusBadge raw={llmStatus} />
            </div>
            <div className="copilot-drawer__header-right">
              {/* Load model button */}
              <button
                onClick={handleLoadModel}
                disabled={loadingModel || llmStatus === 'loading'}
                className="copilot-drawer__btn copilot-drawer__btn--load"
                title="Load GGUF model file"
              >
                {loadingModel || llmStatus === 'loading'
                  ? <Loader size={13} className="copilot-spin" />
                  : <FolderOpen size={13} />}
              </button>
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="copilot-drawer__btn"
                title={isMinimized ? 'Expand' : 'Minimize'}
              >
                {isMinimized ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="copilot-drawer__btn"
                title="Close"
              >
                <X size={14} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Message list */}
              <div className="copilot-drawer__messages">
                {messages.map((msg, i) => (
                  <MessageBubble key={i} msg={msg} />
                ))}
                {sending && (
                  <div className="copilot-typing">
                    <span /><span /><span />
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <form onSubmit={handleSend} className="copilot-drawer__input-row">
                <input
                  id="copilot-chat-input"
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask Co-Pilot anything…"
                  disabled={sending}
                  className="copilot-drawer__input"
                  autoComplete="off"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || sending}
                  className="copilot-drawer__send"
                  title="Send"
                >
                  <Send size={15} />
                </button>
              </form>
            </>
          )}
        </div>
      )}
    </>
  );
}
