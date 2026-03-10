'use client';

import { useEffect, useRef } from 'react';
import { listen } from '@tauri-apps/api/event';

export interface LlmInsight {
  tag: 'advice' | 'alert' | 'chat';
  title: string;
  body: string;
}

interface Props {
  onInsight: (insight: LlmInsight) => void;
}

/**
 * Invisible component — just wires up the Tauri `llm_insight` event listener
 * and calls `onInsight` whenever the Rust backend emits a new LLM payload.
 */
export default function LlmEventBridge({ onInsight }: Props) {
  const onInsightRef = useRef(onInsight);
  useEffect(() => { onInsightRef.current = onInsight; }, [onInsight]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    const setup = async () => {
      unlisten = await listen<LlmInsight>('llm_insight', (event) => {
        onInsightRef.current(event.payload);
      });
    };

    setup();
    return () => { if (unlisten) unlisten(); };
  }, []);

  return null;
}
