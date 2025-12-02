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
    <header class="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div class="container mx-auto flex h-16 items-center justify-between px-4">
        <div class="flex items-center gap-6">
          <!-- Logo -->
          <NuxtLink to="/" class="flex items-center gap-2 font-semibold text-foreground hover:text-primary transition-colors">
            <span class="text-xl">🔧</span>
            <span class="hidden sm:inline">Hydraulic Diagnostics</span>
          </NuxtLink>

          <!-- Navigation -->
          <nav class="hidden md:flex items-center gap-6" aria-label="Main navigation">
            <NuxtLink
              to="/dashboard"
              class="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              active-class="text-foreground"
            >
              Dashboard
            </NuxtLink>
            <NuxtLink
              to="/systems"
              class="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
              active-class="text-foreground"
            >
              Systems
            </NuxtLink>
          </nav>
        </div>

        <!-- Right side actions -->
        <div class="flex items-center gap-4">
          <!-- User menu placeholder -->
          <div class="text-sm text-muted-foreground">
            <!-- Add user menu here -->
          </div>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main id="main-content" class="container mx-auto px-4 py-6">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
// Default layout with header and navigation
// Used for authenticated pages (dashboard, systems, etc.)
</script>
