import { useState, useEffect } from "react";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

export function UpdateBanner() {
  const [updateAvailable, setUpdateAvailable] = useState<{
    version: string;
    body: string;
  } | null>(null);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState("");
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // Check after a short delay so the app loads first
    const timer = setTimeout(async () => {
      try {
        const update = await check();
        if (!cancelled && update?.available) {
          setUpdateAvailable({
            version: update.version,
            body: update.body ?? "",
          });
        }
      } catch (e) {
        console.warn("Update check failed:", e);
      }
    }, 3000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const handleInstall = async () => {
    setInstalling(true);
    try {
      const update = await check();
      if (!update?.available) return;

      setProgress("Downloading...");
      await update.downloadAndInstall((event) => {
        if (event.event === "Started" && event.data.contentLength) {
          setProgress(
            `Downloading (${(event.data.contentLength / 1024 / 1024).toFixed(1)} MB)...`
          );
        } else if (event.event === "Finished") {
          setProgress("Installing...");
        }
      });
      setProgress("Restarting...");
      await relaunch();
    } catch (e) {
      console.error("Update failed:", e);
      setProgress("Update failed. Try again later.");
      setInstalling(false);
    }
  };

  if (!updateAvailable || dismissed) return null;

  return (
    <div className="bg-accent-purple/15 border border-accent-purple/30 rounded-lg px-3 py-2 mx-2 mt-2 flex items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-sm min-w-0">
        <span className="text-accent-purple font-medium shrink-0">
          v{updateAvailable.version} available
        </span>
        {installing && (
          <span className="text-text-muted text-xs">{progress}</span>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {!installing && (
          <>
            <button
              onClick={handleInstall}
              className="text-xs bg-accent-purple/20 hover:bg-accent-purple/30 text-accent-purple px-2.5 py-1 rounded transition-colors cursor-pointer"
            >
              Update & restart
            </button>
            <button
              onClick={() => setDismissed(true)}
              className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer px-1"
            >
              Later
            </button>
          </>
        )}
      </div>
    </div>
  );
}
