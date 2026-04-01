let loadPromise: Promise<void> | null = null;

declare global {
  interface Window {
    Twitch: any;
  }
}

export function loadTwitchApi(): Promise<void> {
  if (window.Twitch?.Player) return Promise.resolve();
  if (loadPromise) return loadPromise;

  loadPromise = new Promise<void>((resolve) => {
    const script = document.createElement("script");
    script.src = "https://player.twitch.tv/js/embed/v1.js";
    script.onload = () => resolve();
    document.head.appendChild(script);
  });

  return loadPromise;
}
