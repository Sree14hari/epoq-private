"use client";

import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { invoke } from "@tauri-apps/api/core";
import { QRCodeSVG } from "qrcode.react";
import { Smartphone } from "lucide-react";

export default function QRCodeConnect() {
  const [isOpen, setIsOpen] = useState(false);
  const [connectionDetails, setConnectionDetails] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleOpen = async () => {
    setIsOpen(true);
    if (!connectionDetails) {
      setLoading(true);
      setError(null);
      try {
        const details = await invoke<string>("get_connection_details");
        setConnectionDetails(details);
      } catch (e: any) {
        setError(e.toString());
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <>
      <button
        onClick={handleOpen}
        className="flex items-center gap-2 px-4 py-2 bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 rounded-full text-sm font-medium text-emerald-400 transition-colors mr-2"
      >
        <Smartphone className="w-4 h-4" />
        Connect Mobile
      </button>

      {isOpen && mounted && createPortal(
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="relative w-full max-w-sm rounded-2xl bg-zinc-900 border border-zinc-800 p-6 shadow-2xl flex flex-col items-center text-center"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-xl font-bold tracking-tight text-white mb-2">
              Connect Mobile App
            </h2>
            <p className="text-sm text-zinc-400 mb-6">
              Scan this QR code with the EPOQ Android app to monitor training.
            </p>

            {loading && (
              <div className="w-48 h-48 flex items-center justify-center">
                <span className="inline-block w-8 h-8 border-4 border-zinc-800 border-t-emerald-500 rounded-full animate-spin" />
              </div>
            )}

            {error && (
              <div className="w-full flex items-center justify-center p-4 text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl mb-6 text-sm">
                Error getting connection details: {error}
              </div>
            )}

            {connectionDetails && !loading && !error && (
              <div className="bg-white p-4 rounded-xl shadow-inner mb-6">
                <QRCodeSVG
                  value={connectionDetails}
                  size={200}
                  level="H"
                  includeMargin={false}
                  fgColor="#000000"
                  bgColor="#ffffff"
                />
              </div>
            )}

            {connectionDetails && !loading && !error && (
              <div className="w-full bg-zinc-950 rounded-lg p-3 text-xs text-zinc-500 mb-6 font-mono break-all text-left border border-zinc-800">
                {connectionDetails}
              </div>
            )}

            <button
              onClick={() => setIsOpen(false)}
              className="w-full rounded-xl border border-zinc-800 bg-zinc-800 hover:bg-zinc-700 py-3 text-sm font-medium text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
