/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "mt1.google.com",
      },
    ],
  },
};

export default nextConfig;
