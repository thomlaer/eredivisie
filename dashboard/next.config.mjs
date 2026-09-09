/** @type {import('next').NextConfig} */
const nextConfig = {
  poweredByHeader: false,
  agentRules: false,
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
