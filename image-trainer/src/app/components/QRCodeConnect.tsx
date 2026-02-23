"use client";

import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { QRCodeSVG } from "qrcode.react";
import { Smartphone } from "lucide-react";

export default function QRCodeConnect() {
  const [isOpen, setIsOpen] = useState(false);
  const [connectionDetails, setConnectionDetails] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        className="flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/20 px-4 py-2 text-sm font-medium text-blue-300 transition-colors"
      >
        <Smartphone className="w-4 h-4" />
        Connect Mobile
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="relative w-full max-w-sm rounded-2xl bg-zinc-900 border border-zinc-700 p-6 shadow-2xl flex flex-col items-center text-center"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-xl font-semibold text-white mb-2">
              Connect Mobile App
            </h2>
            <p className="text-sm text-zinc-400 mb-6">
              Scan this QR code with the EPOQ Android app to monitor training.
            </p>

            {loading && (
              <div className="w-48 h-48 flex items-center justify-center">
                <span className="inline-block w-8 h-8 border-4 border-zinc-600 border-t-blue-500 rounded-full animate-spin" />
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
              <div className="w-full bg-zinc-800 rounded-lg p-3 text-xs text-zinc-400 mb-6 font-mono break-all text-left">
                {connectionDetails}
              </div>
            )}

            <button
              onClick={() => setIsOpen(false)}
              className="w-full rounded-xl border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 py-3 text-sm font-medium text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}
