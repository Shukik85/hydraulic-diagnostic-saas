<script setup lang="ts">
import { ref } from 'vue';

const isChatOpen = ref(false);

const openChat = () => {
  isChatOpen.value = true;
  // TODO: Implement chat modal
  console.log('Chat modal opening...');
};
</script>

<template>
  <div class="min-h-screen bg-background">
    <!-- Skip to main content link for accessibility -->
    <a
      href="#main-content"
      class="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md"
    >
      Skip to main content
    </a>

    <!-- Header -->
    <header class="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-lg supports-[backdrop-filter]:bg-background/60">
      <div class="container mx-auto flex h-16 items-center justify-between gap-4 px-4">
        <div class="flex items-center gap-6">
          <!-- Logo -->
          <NuxtLink
            to="/"
            class="flex items-center gap-2 font-semibold text-foreground transition-colors hover:text-primary"
          >
            <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-600">
              <Icon name="heroicons:wrench-screwdriver" class="h-5 w-5 text-white" />
            </div>
            <span class="hidden sm:inline text-lg">Hydraulic Diagnostics</span>
          </NuxtLink>

          <!-- Navigation -->
          <nav class="hidden md:flex items-center gap-1" aria-label="Main navigation">
            <NuxtLink
              to="/dashboard"
              class="rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              active-class="bg-secondary text-foreground"
            >
              Dashboard
            </NuxtLink>
            <NuxtLink
              to="/systems"
              class="rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              active-class="bg-secondary text-foreground"
            >
              Systems
            </NuxtLink>
            <NuxtLink
              to="/diagnosis"
              class="rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              active-class="bg-secondary text-foreground"
            >
              Diagnosis
            </NuxtLink>
          </nav>
        </div>

        <!-- Right side actions -->
        <div class="flex items-center gap-2">
          <!-- Chat/Search Button -->
          <ChatButton @open-chat="openChat" />

          <!-- Language Switcher -->
          <LanguageSwitcher />

          <!-- Theme Switcher -->
          <ThemeSwitcher />

          <!-- User Menu (placeholder) -->
          <button
            type="button"
            aria-label="User menu"
            class="flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-background transition-all hover:bg-secondary hover:border-primary/50"
          >
            <Icon name="heroicons:user-circle" class="h-5 w-5 text-muted-foreground" />
          </button>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main id="main-content" class="container mx-auto px-4 py-6">
      <slot />
    </main>

    <!-- TODO: Chat Modal Component -->
    <!-- <ChatModal v-if="isChatOpen" @close="isChatOpen = false" /> -->
  </div>
</template>
