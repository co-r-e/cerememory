import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Keep local runs quiet by default; longer test times are acceptable here.
    fileParallelism: false,
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    maxWorkers: 1,
    minWorkers: 1,
    maxConcurrency: 1,
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
    },
  },
});
