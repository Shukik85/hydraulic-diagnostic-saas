import { useState } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import {
  Play,
  Download,
  Share2,
  AlertTriangle,
  CheckCircle,
  Clock,
  TrendingUp,
  FileText,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Progress } from "../ui/progress";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Label } from "../ui/label";

const diagnosticSessions = [
  {
    id: "DIAG-2384",
    equipment: "HYD-002",
    status: "completed",
    severity: "warning",
    date: "2024-10-21 14:32",
    duration: "3m 24s",
    issues: 3,
    recommendations: 5,
  },
  {
    id: "DIAG-2383",
    equipment: "HYD-001",
    status: "completed",
    severity: "success",
    date: "2024-10-21 09:15",
    duration: "2m 18s",
    issues: 0,
    recommendations: 2,
  },
  {
    id: "DIAG-2382",
    equipment: "HYD-004",
    status: "completed",
    severity: "error",
    date: "2024-10-20 16:45",
    duration: "4m 52s",
    issues: 8,
    recommendations: 12,
  },
];

const radarData = [
  { subject: "Pressure", A: 85, fullMark: 100 },
  { subject: "Temperature", A: 92, fullMark: 100 },
  { subject: "Flow Rate", A: 78, fullMark: 100 },
  { subject: "Vibration", A: 88, fullMark: 100 },
  { subject: "Noise Level", A: 95, fullMark: 100 },
  { subject: "Efficiency", A: 82, fullMark: 100 },
];

const timeSeriesData = [
  { time: "10:00", value: 145, threshold: 150 },
  { time: "10:30", value: 148, threshold: 150 },
  { time: "11:00", value: 152, threshold: 150 },
  { time: "11:30", value: 156, threshold: 150 },
  { time: "12:00", value: 159, threshold: 150 },
  { time: "12:30", value: 162, threshold: 150 },
  { time: "13:00", value: 165, threshold: 150 },
];

export function DiagnosticsPage() {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedEquipment, setSelectedEquipment] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const handleRunDiagnostic = () => {
    setIsRunning(true);
    setTimeout(() => {
      setIsRunning(false);
      setIsDialogOpen(false);
    }, 3000);
  };

  return (
    <div className="space-y-6 p-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="mb-2">Diagnostics</h1>
          <p className="text-muted-foreground">
            Automated equipment diagnostics and anomaly detection
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Play className="mr-2 h-4 w-4" />
              Run Diagnostic
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Run Equipment Diagnostic</DialogTitle>
              <DialogDescription>
                Select equipment and configure diagnostic parameters
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label>Equipment</Label>
                <Select value={selectedEquipment} onValueChange={setSelectedEquipment}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select equipment" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hyd-001">HYD-001 - Hydraulic Press A1</SelectItem>
                    <SelectItem value="hyd-002">HYD-002 - Hydraulic Pump B2</SelectItem>
                    <SelectItem value="hyd-003">HYD-003 - Hydraulic Lift C3</SelectItem>
                    <SelectItem value="hyd-004">HYD-004 - Hydraulic System D4</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Diagnostic Type</Label>
                <Select defaultValue="full">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="full">Full System Check</SelectItem>
                    <SelectItem value="quick">Quick Scan</SelectItem>
                    <SelectItem value="sensors">Sensor Validation</SelectItem>
                    <SelectItem value="performance">Performance Analysis</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {isRunning && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Running diagnostic...</span>
                    <span>67%</span>
                  </div>
                  <Progress value={67} className="h-2" />
                </div>
              )}
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleRunDiagnostic} disabled={!selectedEquipment || isRunning}>
                {isRunning ? "Running..." : "Start Diagnostic"}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Latest Diagnostic Result */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="border-yellow-500/50 bg-yellow-500/5">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="h-12 w-12 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <AlertTriangle className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
                </div>
                <div>
                  <CardTitle>Latest Diagnostic - HYD-002</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Session DIAG-2384 • Completed 15 minutes ago
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Export PDF
                </Button>
                <Button variant="outline" size="sm">
                  <Share2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Issues Found</p>
                      <p className="text-2xl font-bold">3</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-yellow-600" />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Recommendations</p>
                      <p className="text-2xl font-bold">5</p>
                    </div>
                    <FileText className="h-8 w-8 text-blue-600" />
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Overall Score</p>
                      <p className="text-2xl font-bold">76%</p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-green-600" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Charts */}
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">System Health Radar</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="var(--border)" />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar
                        name="Health"
                        dataKey="A"
                        stroke="var(--primary)"
                        fill="var(--primary)"
                        fillOpacity={0.3}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Pressure Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={timeSeriesData}>
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
                      <Line
                        type="monotone"
                        dataKey="threshold"
                        stroke="var(--destructive)"
                        strokeDasharray="5 5"
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="var(--primary)"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Issues & Recommendations */}
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Detected Issues</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                    <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">High Pressure Detected</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Current: 165 PSI (Normal: 145-150 PSI)
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">Temperature Fluctuation</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Variance: ±5°C over 2 hours
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">Sensor Calibration Due</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Last calibrated 89 days ago
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Recommendations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex gap-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">Inspect Pressure Relief Valve</p>
                      <p className="text-xs text-muted-foreground mt-1">Priority: High</p>
                    </div>
                  </div>
                  <div className="flex gap-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">Check Hydraulic Lines</p>
                      <p className="text-xs text-muted-foreground mt-1">Priority: High</p>
                    </div>
                  </div>
                  <div className="flex gap-3 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">Schedule Sensor Calibration</p>
                      <p className="text-xs text-muted-foreground mt-1">Priority: Medium</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Diagnostic History */}
      <Card>
        <CardHeader>
          <CardTitle>Diagnostic History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {diagnosticSessions.map((session, index) => (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="flex items-center justify-between p-4 rounded-lg bg-muted/50 hover:bg-muted transition-colors-smooth cursor-pointer"
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`h-10 w-10 rounded-lg flex items-center justify-center ${
                      session.severity === "success"
                        ? "bg-green-500/20"
                        : session.severity === "warning"
                        ? "bg-yellow-500/20"
                        : "bg-red-500/20"
                    }`}
                  >
                    {session.severity === "success" ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <AlertTriangle
                        className={`h-5 w-5 ${
                          session.severity === "warning"
                            ? "text-yellow-600"
                            : "text-red-600"
                        }`}
                      />
                    )}
                  </div>
                  <div>
                    <p className="font-medium">
                      {session.id} - {session.equipment}
                    </p>
                    <div className="flex items-center gap-3 text-sm text-muted-foreground mt-1">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {session.date}
                      </span>
                      <span>•</span>
                      <span>{session.duration}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Issues</p>
                    <p className="text-lg font-bold">{session.issues}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground">Actions</p>
                    <p className="text-lg font-bold">{session.recommendations}</p>
                  </div>
                  <Button variant="ghost" size="sm">
                    <FileText className="h-4 w-4" />
                  </Button>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
