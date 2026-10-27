// First launch (empty data dir), the rail, the tool pages and the error banner.
import { test, expect } from "../fixtures";

test.describe("first launch", () => {
  test("shows the welcome page when nothing is analysed yet", async ({ app }) => {
    const { page } = app;
    await expect(page.getByRole("heading", { name: "Welcome to Spaghetti Lab" })).toBeVisible();
    for (const step of ["Add a match", "Let it read the game", "Review"]) {
      await expect(page.getByText(step, { exact: true })).toBeVisible();
    }
    await expect(page.getByRole("button", { name: /Drop a match video here/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Other things you can do" })).toBeVisible();
    await expect(app.rail("Library")).toHaveAttribute("aria-current", "page");
  });

  test("welcome tool cards open their pages", async ({ app }) => {
    const { page } = app;
    const cases: [RegExp, string][] = [
      [/^Annotate a VOD/, "Annotate a video"],
      [/^Split a tournament VOD/, "Split a tournament VOD"],
      [/^Export Strive replays/, "Export Strive replays"],
    ];
    for (const [card, heading] of cases) {
      await app.rail("Library").click();
      await page.getByRole("button", { name: card }).click();
      await expect(page.getByRole("heading", { level: 1 })).toHaveText(heading);
      await page.getByRole("main").getByRole("button", { name: "Library", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Welcome to Spaghetti Lab" })).toBeVisible();
    }
    // the drop zone opens the analyse page
    await page.getByRole("button", { name: /Drop a match video here/ }).click();
    await expect(page.getByRole("heading", { name: "Analyse a match" })).toBeVisible();
  });

  test("open a shared analysis: a cancelled dialog changes nothing", async ({ app }) => {
    const { page } = app;
    await app.rail("Library").click();
    await page.getByRole("button", { name: /^Open a shared analysis/ }).click();
    const opens = await app.calls("plugin:dialog|open");
    expect(opens).toHaveLength(1);
    expect(JSON.stringify(opens[0].args)).toContain("spagz");
    await expect(page.getByRole("heading", { name: "Welcome to Spaghetti Lab" })).toBeVisible();
  });
});

test.describe("rail", () => {
  const pages: [string, string][] = [
    ["Analyse", "Analyse a match"],
    ["Annotate", "Annotate a video"],
    ["Split VOD", "Split a tournament VOD"],
    ["Strive", "Export Strive replays"],
  ];

  for (const [label, heading] of pages) {
    test(`${label} opens its page and marks itself active`, async ({ app }) => {
      await app.rail(label).click();
      await expect(app.page.getByRole("heading", { level: 1 })).toHaveText(heading);
      await expect(app.rail(label)).toHaveAttribute("aria-current", "page");
      await expect(app.rail("Library")).not.toHaveAttribute("aria-current", "page");
    });
  }

  test("the logo goes home", async ({ app }) => {
    await app.rail("Strive").click();
    await app.page.getByRole("navigation").getByRole("button", { name: "Spaghetti Lab" }).click();
    await expect(app.rail("Library")).toHaveAttribute("aria-current", "page");
  });

  test("Open asks for a .spag / .spagz file", async ({ app }) => {
    await app.rail("Open").click();
    const [call] = await app.calls("plugin:dialog|open");
    expect(call.args.options.filters[0].extensions).toEqual(["spag", "spagz"]);
  });

  test("Support opens the donation page in the browser", async ({ app }) => {
    await app.rail("Support").click();
    const [call] = await app.calls("plugin:shell|open");
    expect(call.args.path).toMatch(/^https:\/\//);
  });
});

test.describe("drag and drop", () => {
  test("dragging over the window highlights the drop zone", async ({ app }) => {
    const { page } = app;
    await app.rail("Library").click();
    await app.dragOver(["C:/somewhere/match.mp4"]);
    await expect(page.getByRole("button", { name: /Drop it to start the analysis/ })).toBeVisible();
    await app.dragLeave();
    await expect(page.getByRole("button", { name: /Drop a match video here/ })).toBeVisible();
  });

  test("an unsupported file shows the error banner, which can be dismissed", async ({ app }) => {
    const { page } = app;
    await app.dropFiles(["C:/somewhere/notes.txt"]);
    const banner = page.getByText("Drop a match video (MP4, MKV, AVI) or a .spag / .spagz file.");
    await expect(banner).toBeVisible();
    await page.getByRole("button", { name: "Dismiss" }).click();
    await expect(banner).toHaveCount(0);
  });
});
