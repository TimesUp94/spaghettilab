export type VideoSourceType = "file" | "youtube" | "twitch";

export function detectVideoSource(url: string): { type: VideoSourceType; id?: string } {
  const ytMatch = url.match(
    /(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/|live\/))([\w-]+)/
  );
  if (ytMatch) return { type: "youtube", id: ytMatch[1] };

  const twitchMatch = url.match(/twitch\.tv\/videos\/(\d+)/);
  if (twitchMatch) return { type: "twitch", id: twitchMatch[1] };

  return { type: "file" };
}
