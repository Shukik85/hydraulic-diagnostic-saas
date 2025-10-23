import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";

const stats = [
  {
    title: "Active Equipment",
    value: "142",
    change: "+12%",
    trend: "up",
    icon: Activity,
  },
  {
    title: "Anomalies Detected",
    value: "8",
    change: "-23%",
    trend: "down",
    icon: AlertTriangle,
  },
  {
    title: "System Health",
    value: "96.4%",
    change: "+2.1%",
    trend: "up",
    icon: CheckCircle,
  },
  {
    title: "Avg Response Time",
    value: "1.2s",
    change: "-0.3s",
    trend: "down",
    icon: Clock,
  },
];

const chartData = [
  { time: "00:00", pressure: 145, temperature: 68, flow: 22 },
  { time: "04:00", pressure: 148, temperature: 71, flow: 24 },
  { time: "08:00", pressure: 152, temperature: 74, flow: 26 },
  { time: "12:00", pressure: 149, temperature: 73, flow: 25 },
  { time: "16:00", pressure: 146, temperature: 70, flow: 23 },
  { time: "20:00", pressure: 144, temperature: 69, flow: 22 },
];

const equipmentData = [
  { name: "HYD-001", status: "online", health: 98, location: "Building A" },
  { name: "HYD-002", status: "warning", health: 76, location: "Building B" },
  { name: "HYD-003", status: "online", health: 95, location: "Building A" },
  { name: "HYD-004", status: "error", health: 45, location: "Building C" },
  { name: "HYD-005", status: "online", health: 92, location: "Building B" },
];

const recentActivity = [
  {
    id: 1,
    type: "anomaly",
    message: "High pressure detected on HYD-002",
    time: "2 min ago",
  },
  {
    id: 2,
    type: "success",
    message: "Diagnostic report #2384 completed",
    time: "15 min ago",
  },
  {
    id: 3,
    type: "info",
    message: "Sensor calibration scheduled for HYD-001",
    time: "1 hour ago",
  },
  {
    id: 4,
    type: "warning",
    message: "Temperature threshold exceeded on HYD-004",
    time: "2 hours ago",
  },
];

export function DashboardPage() {
  return (
    <div className="space-y-6 p-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="mb-2">Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time monitoring and system overview
          </p>
        </div>
        <Button>
          <Activity className="mr-2 h-4 w-4" />
          Run Diagnostics
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card className="transition-all duration-300 hover:shadow-lg hover:-translate-y-1">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.title}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
                <p
                  className={`flex items-center text-xs mt-1 ${
                    stat.trend === "up" ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {stat.trend === "up" ? (
                    <TrendingUp className="mr-1 h-3 w-3" />
                  ) : (
                    <TrendingDown className="mr-1 h-3 w-3" />
                  )}
                  {stat.change} from last week
                </p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.4 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Pressure & Temperature Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient
                      id="colorPressure"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor="var(--primary)"
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor="var(--primary)"
                        stopOpacity={0}
                      />
                    </linearGradient>
                    <linearGradient
                      id="colorTemp"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor="var(--secondary)"
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor="var(--secondary)"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--card)",
                      border: "1px solid var(--border)",
                      borderRadius: "8px",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="pressure"
                    stroke="var(--primary)"
                    fillOpacity={1}
                    fill="url(#colorPressure)"
                    strokeWidth={2}
                  />
                  <Area
                    type="monotone"
                    dataKey="temperature"
                    stroke="var(--secondary)"
                    fillOpacity={1}
                    fill="url(#colorTemp)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.5 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Equipment Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {equipmentData.map((equipment, index) => (
                  <motion.div
                    key={equipment.name}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.6 + index * 0.05 }}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors-smooth cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`h-2 w-2 rounded-full ${
                          equipment.status === "online"
                            ? "bg-green-500"
                            : equipment.status === "warning"
                            ? "bg-yellow-500"
                            : "bg-red-500"
                        }`}
                      />
                      <div>
                        <p className="font-medium">{equipment.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {equipment.location}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{equipment.health}%</p>
                      <div className="w-20 mt-1">
                        <Progress value={equipment.health} className="h-1" />
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentActivity.map((activity, index) => (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.7 + index * 0.05 }}
                  className="flex items-start gap-3 p-3 rounded-lg hover:bg-muted/50 transition-colors-smooth cursor-pointer"
                >
                  <div
                    className={`mt-0.5 h-2 w-2 rounded-full ${
                      activity.type === "success"
                        ? "bg-green-500"
                        : activity.type === "warning"
                        ? "bg-yellow-500"
                        : activity.type === "error"
                        ? "bg-red-500"
                        : "bg-blue-500"
                    }`}
                  />
                  <div className="flex-1">
                    <p className="text-sm">{activity.message}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {activity.time}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
