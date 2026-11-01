// PM2: pm2 start server/opener_service/ecosystem.config.js (from the repo root, OPENERS_SECRET in the environment)
module.exports = {
  apps: [{
    name: "opener-service",
    cwd: __dirname + "/../..",
    script: ".venv/bin/uvicorn",
    args: "--factory server.opener_service.app:make_app --host 127.0.0.1 --port 10490 --workers 1",
    interpreter: "none",
    max_memory_restart: "4G",
    env: {
      OPENERS_SECRET: process.env.OPENERS_SECRET,
      GGST_MODEL_DIR: "/opt/replanal-models",
      GGST_DEVICE: "cpu",
      OPENERS_WORKDIR: "/var/lib/opener-service",
      OPENERS_MAX_CONCURRENT: "1",
      OPENERS_MAX_QUEUE: "8",
      OPENERS_PER_OWNER: "1",
      OPENERS_THREADS: "2",
      OPENERS_LIGHT: "1",
      OPENERS_URL_HOSTS: "fra1.digitaloceanspaces.com",
    },
  }],
};
