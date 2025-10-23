import { useState } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import {
  Plus,
  Search,
  Filter,
  Grid3x3,
  List,
  MapPin,
  Activity,
  Clock,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Progress } from "../ui/progress";
import { ImageWithFallback } from "../figma/ImageWithFallback";

const equipmentData = [
  {
    id: "HYD-001",
    name: "Hydraulic Press A1",
    location: "Building A - Floor 1",
    status: "online",
    health: 98,
    lastMaintenance: "2 days ago",
    sensors: 12,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
  {
    id: "HYD-002",
    name: "Hydraulic Pump B2",
    location: "Building B - Floor 2",
    status: "warning",
    health: 76,
    lastMaintenance: "1 week ago",
    sensors: 8,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
  {
    id: "HYD-003",
    name: "Hydraulic Lift C3",
    location: "Building A - Floor 3",
    status: "online",
    health: 95,
    lastMaintenance: "3 days ago",
    sensors: 10,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
  {
    id: "HYD-004",
    name: "Hydraulic System D4",
    location: "Building C - Floor 1",
    status: "error",
    health: 45,
    lastMaintenance: "2 weeks ago",
    sensors: 15,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
  {
    id: "HYD-005",
    name: "Hydraulic Cylinder E5",
    location: "Building B - Floor 1",
    status: "online",
    health: 92,
    lastMaintenance: "5 days ago",
    sensors: 6,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
  {
    id: "HYD-006",
    name: "Hydraulic Motor F6",
    location: "Building A - Floor 2",
    status: "maintenance",
    health: 88,
    lastMaintenance: "Today",
    sensors: 9,
    image: "https://images.unsplash.com/photo-1723532773642-2cd95209c3b3?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxpbmR1c3RyaWFsJTIwbWFjaGluZXJ5JTIwaHlkcmF1bGljfGVufDF8fHx8MTc2MTA1MjQzNHww&ixlib=rb-4.1.0&q=80&w=400",
  },
];

export function EquipmentListPage() {
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchQuery, setSearchQuery] = useState("");

  const getStatusBadge = (status: string) => {
    const variants = {
      online: { variant: "default" as const, label: "Online", color: "bg-green-500" },
      warning: { variant: "secondary" as const, label: "Warning", color: "bg-yellow-500" },
      error: { variant: "destructive" as const, label: "Error", color: "bg-red-500" },
      maintenance: { variant: "outline" as const, label: "Maintenance", color: "bg-blue-500" },
    };
    return variants[status as keyof typeof variants] || variants.online;
  };

  return (
    <div className="space-y-6 p-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="mb-2">Equipment</h1>
          <p className="text-muted-foreground">
            Manage and monitor all hydraulic equipment
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Add Equipment
        </Button>
      </div>

      {/* Filter Bar */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex-1 max-w-md">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search equipment..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Select defaultValue="all">
                <SelectTrigger className="w-[150px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="online">Online</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="maintenance">Maintenance</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="all">
                <SelectTrigger className="w-[150px]">
                  <SelectValue placeholder="Location" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Locations</SelectItem>
                  <SelectItem value="building-a">Building A</SelectItem>
                  <SelectItem value="building-b">Building B</SelectItem>
                  <SelectItem value="building-c">Building C</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
              </Button>
              <div className="flex border border-border rounded-lg">
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid3x3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setViewMode("list")}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Equipment Grid/List */}
      {viewMode === "grid" ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {equipmentData.map((equipment, index) => {
            const statusInfo = getStatusBadge(equipment.status);
            return (
              <motion.div
                key={equipment.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1 cursor-pointer">
                  <div className="relative h-48 overflow-hidden">
                    <ImageWithFallback
                      src={equipment.image}
                      alt={equipment.name}
                      className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
                    />
                    <div className="absolute top-3 right-3">
                      <Badge variant={statusInfo.variant}>
                        {statusInfo.label}
                      </Badge>
                    </div>
                    <div className="absolute top-3 left-3">
                      <div className={`h-3 w-3 rounded-full ${statusInfo.color} animate-pulse`} />
                    </div>
                  </div>
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div>
                        <h3 className="mb-1">{equipment.name}</h3>
                        <p className="text-sm text-muted-foreground flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {equipment.location}
                        </p>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Health</span>
                          <span className="font-medium">{equipment.health}%</span>
                        </div>
                        <Progress value={equipment.health} className="h-2" />
                      </div>
                      <div className="flex items-center justify-between text-sm pt-2 border-t border-border">
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Activity className="h-3 w-3" />
                          <span>{equipment.sensors} sensors</span>
                        </div>
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          <span>{equipment.lastMaintenance}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="divide-y divide-border">
              {equipmentData.map((equipment, index) => {
                const statusInfo = getStatusBadge(equipment.status);
                return (
                  <motion.div
                    key={equipment.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="p-4 hover:bg-muted/50 transition-colors-smooth cursor-pointer"
                  >
                    <div className="flex items-center gap-4">
                      <div className="relative h-16 w-16 rounded-lg overflow-hidden flex-shrink-0">
                        <ImageWithFallback
                          src={equipment.image}
                          alt={equipment.name}
                          className="w-full h-full object-cover"
                        />
                        <div
                          className={`absolute top-1 left-1 h-2 w-2 rounded-full ${statusInfo.color}`}
                        />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="truncate">{equipment.name}</h4>
                          <Badge variant={statusInfo.variant} className="flex-shrink-0">
                            {statusInfo.label}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {equipment.location}
                        </p>
                      </div>
                      <div className="hidden md:flex items-center gap-8 flex-shrink-0">
                        <div className="text-center">
                          <p className="text-xs text-muted-foreground mb-1">Health</p>
                          <p className="font-medium">{equipment.health}%</p>
                        </div>
                        <div className="text-center">
                          <p className="text-xs text-muted-foreground mb-1">Sensors</p>
                          <p className="font-medium">{equipment.sensors}</p>
                        </div>
                        <div className="text-center min-w-[100px]">
                          <p className="text-xs text-muted-foreground mb-1">Last Maintenance</p>
                          <p className="font-medium text-sm">{equipment.lastMaintenance}</p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
