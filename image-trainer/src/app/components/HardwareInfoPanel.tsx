"use client";

import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

// Shape of the JSON returned by check_gpu.py
interface GpuInfo {
    python_version: string;
    torch_version: string;
    cuda_available: boolean;
    cuda_version: string | null;
    device_count: number;
    device_name: string | null;
}

type PanelState =
    | { status: "loading" }
    | { status: "gpu"; info: GpuInfo }
    | { status: "cpu"; info: GpuInfo }
    | { status: "error"; message: string };

export default function HardwareInfoPanel() {
    const [state, setState] = useState<PanelState>({ status: "loading" });

    useEffect(() => {
        invoke<string>("check_gpu")
            .then((raw) => {
                const info: GpuInfo = JSON.parse(raw);
                setState(info.cuda_available ? { status: "gpu", info } : { status: "cpu", info });
            })
            .catch((err) => {
                setState({ status: "error", message: String(err) });
            });
    }, []);

    if (state.status === "loading") {
        return (
            <div className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-2 text-sm text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400 animate-pulse">
                <span className="h-2 w-2 rounded-full bg-zinc-400"></span>
                Checking hardware…
            </div>
        );
    }

    if (state.status === "error") {
        return (
            <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-600 dark:border-red-800 dark:bg-red-950 dark:text-red-400">
                <span>⚠️</span>
                <span>Could not detect hardware — {state.message}</span>
            </div>
        );
    }

    if (state.status === "cpu") {
        return (
            <div className="flex items-center gap-2 rounded-lg border border-yellow-200 bg-yellow-50 px-4 py-2 text-sm text-yellow-700 dark:border-yellow-700 dark:bg-yellow-950 dark:text-yellow-300">
                <span>⚠️</span>
                <span>
                    No GPU detected — Training will run on <strong>CPU</strong> (may be slow)
                </span>
                <span className="ml-auto text-xs text-yellow-500 dark:text-yellow-500">
                    Python {state.info.python_version} · PyTorch {state.info.torch_version}
                </span>
            </div>
        );
    }

    // GPU available
    const { info } = state;
    return (
        <div className="flex items-center gap-3 rounded-lg border border-green-200 bg-green-50 px-4 py-2 text-sm text-green-700 dark:border-green-800 dark:bg-green-950 dark:text-green-300">
            <span>✅</span>
            <span>
                <strong>{info.device_name}</strong>
            </span>
            {info.cuda_version && (
                <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-600 dark:bg-green-900 dark:text-green-400">
                    CUDA {info.cuda_version}
                </span>
            )}
            <span className="ml-auto text-xs text-green-500 dark:text-green-500">
                PyTorch {info.torch_version} · Python {info.python_version}
            </span>
        </div>
    );
}
