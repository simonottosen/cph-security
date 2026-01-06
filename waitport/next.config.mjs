import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const aliasMap = {
  "@": path.resolve(process.cwd(), "src"),
  "@components": path.resolve(process.cwd(), "src/components"),
  "@lib": path.resolve(process.cwd(), "src/lib"),
  "@utils": path.resolve(process.cwd(), "src/utils"),
};

const nextConfig = {
  output: "standalone",
  typescript: {
    ignoreBuildErrors: true,
  },
  turbopack: {
    root: __dirname,
    resolveAlias: aliasMap,
  },
  webpack(config) {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      ...aliasMap,
    };
    return config;
  },
};

export default nextConfig;
