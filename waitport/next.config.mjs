import path from "node:path";
const nextConfig = {
  // ... rest of your configuration
  output: "standalone",
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  webpack(config) {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "@": path.resolve(process.cwd(), "src"),
      "@components": path.resolve(process.cwd(), "src/components"),
      "@lib": path.resolve(process.cwd(), "src/lib"),
      "@utils": path.resolve(process.cwd(), "src/utils"),
    };
    return config;
  },
};

export default nextConfig;
