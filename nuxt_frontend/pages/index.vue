<template>
  <NuxtLayout name="default">
    <div class="space-y-8">
      <!-- Welcome Section -->
      <div class="text-center space-y-4">
        <h1 class="text-4xl font-bold tracking-tight">
          Дашборд системы
        </h1>
        <p class="text-xl text-muted-foreground max-w-2xl mx-auto">
          Обзор состояния гидравлических систем
        </p>
      </div>

      <!-- Quick Stats -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm transition-all hover:shadow-md">
          <div class="flex flex-col items-center justify-center text-center space-y-2">
            <div class="text-3xl font-bold">245</div>
            <div class="text-sm font-medium text-muted-foreground">Активных датчиков</div>
            <div class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Норма
            </div>
          </div>
        </div>

        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm transition-all hover:shadow-md">
          <div class="flex flex-col items-center justify-center text-center space-y-2">
            <div class="text-3xl font-bold">89</div>
            <div class="text-sm font-medium text-muted-foreground">Систем под наблюдением</div>
            <div class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
              Внимание
            </div>
          </div>
        </div>

        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm transition-all hover:shadow-md">
          <div class="flex flex-col items-center justify-center text-center space-y-2">
            <div class="text-3xl font-bold">12</div>
            <div class="text-sm font-medium text-muted-foreground">Критических ошибок</div>
            <div class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
              Ошибка
            </div>
          </div>
        </div>

        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm transition-all hover:shadow-md">
          <div class="flex flex-col items-center justify-center text-center space-y-2">
            <div class="text-3xl font-bold">98%</div>
            <div class="text-sm font-medium text-muted-foreground">Эффективность</div>
            <div class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Отлично
            </div>
          </div>
        </div>
      </div>

      <!-- Charts Section -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm">
          <h3 class="text-lg font-semibold mb-4">Давление в реальном времени</h3>
          <div class="h-64 bg-muted/50 rounded-lg flex items-center justify-center">
            <p class="text-muted-foreground">График давления (модель данных)</p>
          </div>
        </div>

        <div class="bg-card text-card-foreground rounded-xl border p-6 shadow-sm">
          <h3 class="text-lg font-semibold mb-4">Температура систем</h3>
          <div class="h-64 bg-muted/50 rounded-lg flex items-center justify-center">
            <p class="text-muted-foreground">График температуры (модель данных)</p>
          </div>
        </div>
      </div>

      <!-- Recent Diagnostics Table -->
      <div class="bg-card text-card-foreground rounded-xl border shadow-sm">
        <div class="p-6 border-b">
          <h3 class="text-lg font-semibold">Последние диагностические сессии</h3>
        </div>
        <div class="p-6">
          <div class="overflow-x-auto">
            <table class="w-full">
              <thead>
                <tr class="border-b">
                  <th class="text-left py-3 px-4 font-medium">Оборудование</th>
                  <th class="text-left py-3 px-4 font-medium">Статус</th>
                  <th class="text-left py-3 px-4 font-medium">Время начала</th>
                  <th class="text-left py-3 px-4 font-medium">Длительность</th>
                </tr>
              </thead>
              <tbody>
                <tr class="border-b">
                  <td class="py-3 px-4">Насосная станция #P-234</td>
                  <td class="py-3 px-4">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Завершено
                    </span>
                  </td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">10:45, 15 мая 2025</td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">15 мин</td>
                </tr>
                <tr class="border-b">
                  <td class="py-3 px-4">Гидравлический пресс #HP-89</td>
                  <td class="py-3 px-4">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      Ошибка
                    </span>
                  </td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">09:20, 15 мая 2025</td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">8 мин</td>
                </tr>
                <tr>
                  <td class="py-3 px-4">Система охлаждения #C-12</td>
                  <td class="py-3 px-4">
                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Завершено
                    </span>
                  </td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">14:30, 14 мая 2025</td>
                  <td class="py-3 px-4 text-sm text-muted-foreground">22 мин</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </NuxtLayout>
</template>

<script setup lang="ts">
// Dashboard page - main landing page after login
</script>
