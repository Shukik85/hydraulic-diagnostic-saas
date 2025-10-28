<template>
  <div class="w-full h-full">
    <ResponsiveContainer width="100%" height="100%">
      <BarChart :data="data" :margin="{ top: 20, right: 30, left: 20, bottom: 20 }">
        <CartesianGrid strokeDasharray="3 3" class="stroke-muted" />
        <XAxis
          :dataKey="xKey"
          class="text-muted-foreground"
          :fontSize="12"
          :tickLine="false"
          :axisLine="false"
        />
        <YAxis class="text-muted-foreground" :fontSize="12" :tickLine="false" :axisLine="false" />
        <Tooltip
          :contentStyle="{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          }"
          :labelStyle="{ color: 'hsl(var(--foreground))' }"
        />
        <Bar
          :dataKey="dataKey"
          :fill="color"
          :radius="[4, 4, 0, 0]"
          :animationDuration="1000"
          animationEasing="ease-in-out"
        />
      </BarChart>
    </ResponsiveContainer>
  </div>
</template>

<script setup lang="ts">
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
  data: any[];
  dataKey: string;
  xKey?: string;
  color?: string;
}

withDefaults(defineProps<Props>(), {
  xKey: 'name',
  color: 'hsl(var(--primary))',
});
</script>
