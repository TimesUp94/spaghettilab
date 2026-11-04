// Stops the app instance started by the global setup (and its Python / ffmpeg children), the mock
// server, and removes the test sign-in from the OS credential store.
import { clearTestCredentials, stopApp } from "./app-process";
import { stopMockBackend } from "./mock-lab-backend";

export default async function globalTeardown() {
  if (process.env.E2E_KEEP_APP) return;
  await stopApp();
  await stopMockBackend();
  clearTestCredentials();
}
