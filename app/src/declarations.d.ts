declare module "@tauri-apps/plugin-dialog" {
  interface OpenDialogOptions {
    multiple?: boolean;
    filters?: { name: string; extensions: string[] }[];
    defaultPath?: string;
    directory?: boolean;
    title?: string;
  }
  interface SaveDialogOptions {
    filters?: { name: string; extensions: string[] }[];
    defaultPath?: string;
    title?: string;
  }
  export function open(options?: OpenDialogOptions): Promise<string | string[] | null>;
  export function save(options?: SaveDialogOptions): Promise<string | null>;
}

declare module "@tauri-apps/plugin-shell" {
  export function open(path: string): Promise<void>;
}
