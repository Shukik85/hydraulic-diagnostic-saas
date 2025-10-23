import { defineNuxtConfig } from "nuxt/config";

export default defineNuxtConfig({
  devtools: { enabled: true },
  css: ["~/assets/css/globals.css"],
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
  typescript: {
    strict: true,
    typeCheck: true,
    shim: false,
  },
  modules: ["@pinia/nuxt"], // Убрал @nuxtjs/tailwindcss так как настраиваем Tailwind вручную
  app: {
    head: {
      title: "HydraulicsTell",
      meta: [
        { charset: "utf-8" },
        { name: "viewport", content: "width=device-width, initial-scale=1" },
      ],
    },
  },
  nitro: {
    preset: "node-server",
  },
  build: {
    transpile: [],
  },
  imports: {
    autoImport: true,
  },
});
