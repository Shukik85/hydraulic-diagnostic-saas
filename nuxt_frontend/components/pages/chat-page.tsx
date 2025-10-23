import { useState, useRef, useEffect } from "react";
import { motion } from "motion/react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { Send, Bot, User, Paperclip, Sparkles, FileText } from "lucide-react";
import { ScrollArea } from "../ui/scroll-area";
import { Separator } from "../ui/separator";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  context?: string[];
};

const initialMessages: Message[] = [
  {
    id: 1,
    role: "assistant",
    content:
      "Hello! I'm your AI diagnostic assistant. I can help you with equipment troubleshooting, sensor data analysis, maintenance recommendations, and technical documentation. How can I assist you today?",
    timestamp: new Date(Date.now() - 120000),
  },
];

const sampleContexts = [
  "Equipment Manual - HYD-002",
  "Sensor Calibration Guide",
  "Pressure Threshold Standards",
];

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: messages.length + 2,
        role: "assistant",
        content: generateResponse(input),
        timestamp: new Date(),
        context: sampleContexts,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const generateResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase();
    if (lowerQuery.includes("pressure") || lowerQuery.includes("hyd-002")) {
      return "Based on the sensor data from HYD-002, I've detected elevated pressure readings over the past 6 hours. The current pressure is 165 PSI, which is 10% above the normal operating range of 145-150 PSI.\n\nRecommended actions:\n1. Inspect the pressure relief valve for proper operation\n2. Check for potential blockages in the hydraulic lines\n3. Verify fluid viscosity is within specifications\n\nWould you like me to schedule a maintenance check or run a detailed diagnostic?";
    }
    if (lowerQuery.includes("sensor") || lowerQuery.includes("calibration")) {
      return "I can help you with sensor calibration. The standard calibration procedure involves:\n\n1. Verify baseline readings at zero pressure\n2. Apply known reference pressures (50, 100, 150 PSI)\n3. Record and compare sensor outputs\n4. Adjust calibration factors if deviation exceeds ±2%\n\nWhich equipment would you like to calibrate?";
    }
    return "I understand you're asking about " + query + ". Let me search through the technical documentation and sensor data to provide you with accurate information. Could you provide more specific details about the equipment ID or the type of issue you're experiencing?";
  };

  return (
    <div className="h-[calc(100vh-8rem)] p-6 animate-fade-in">
      <div className="grid lg:grid-cols-[1fr,320px] gap-6 h-full">
        {/* Chat Area */}
        <Card className="flex flex-col">
          <CardHeader className="border-b border-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                  <Bot className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle>AI Diagnostic Assistant</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    RAG-powered technical support
                  </p>
                </div>
              </div>
              <Badge variant="outline" className="gap-1">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                Online
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="flex-1 p-0 overflow-hidden">
            <ScrollArea className="h-full" ref={scrollRef}>
              <div className="p-6 space-y-4">
                {messages.map((message, index) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className={`flex gap-3 ${
                      message.role === "user" ? "flex-row-reverse" : ""
                    }`}
                  >
                    <div
                      className={`h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-gradient-to-br from-primary to-secondary text-white"
                      }`}
                    >
                      {message.role === "user" ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={`flex-1 max-w-[80%] ${
                        message.role === "user" ? "items-end" : ""
                      }`}
                    >
                      <div
                        className={`rounded-2xl p-4 ${
                          message.role === "user"
                            ? "bg-primary text-primary-foreground ml-auto"
                            : "bg-muted"
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        {message.context && (
                          <div className="mt-3 pt-3 border-t border-border/50 space-y-1">
                            <p className="text-xs opacity-70 flex items-center gap-1">
                              <Sparkles className="h-3 w-3" />
                              Sources:
                            </p>
                            {message.context.map((ctx, i) => (
                              <Badge
                                key={i}
                                variant="outline"
                                className="text-xs mr-1"
                              >
                                {ctx}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 px-2">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </motion.div>
                ))}

                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-3"
                  >
                    <div className="h-8 w-8 rounded-full bg-gradient-to-br from-primary to-secondary text-white flex items-center justify-center">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="bg-muted rounded-2xl p-4">
                      <div className="flex gap-1">
                        <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce" />
                        <div
                          className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        />
                        <div
                          className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce"
                          style={{ animationDelay: "0.4s" }}
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </ScrollArea>
          </CardContent>

          <div className="p-4 border-t border-border">
            <div className="flex gap-2">
              <Button variant="outline" size="icon">
                <Paperclip className="h-4 w-4" />
              </Button>
              <Input
                placeholder="Ask about diagnostics, sensors, maintenance..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSend()}
                className="flex-1"
              />
              <Button onClick={handleSend} disabled={!input.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* Context Panel */}
        <Card className="hidden lg:block">
          <CardHeader>
            <CardTitle className="text-base">Session Context</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">Active Equipment</p>
              <div className="space-y-2">
                <div className="p-2 rounded-lg bg-muted/50 text-sm">
                  <p className="font-medium">HYD-002</p>
                  <p className="text-xs text-muted-foreground">
                    Hydraulic Pump B2
                  </p>
                </div>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-sm font-medium mb-2">Recent Topics</p>
              <div className="space-y-1">
                <Badge variant="outline" className="mr-1 mb-1">
                  Pressure Analysis
                </Badge>
                <Badge variant="outline" className="mr-1 mb-1">
                  Sensor Calibration
                </Badge>
                <Badge variant="outline" className="mr-1 mb-1">
                  Maintenance
                </Badge>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-sm font-medium mb-2">Documentation</p>
              <div className="space-y-2">
                <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm cursor-pointer hover:bg-muted transition-colors-smooth">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-xs">Equipment Manual</span>
                </div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm cursor-pointer hover:bg-muted transition-colors-smooth">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-xs">Calibration Guide</span>
                </div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm cursor-pointer hover:bg-muted transition-colors-smooth">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-xs">Safety Standards</span>
                </div>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-sm font-medium mb-2">Quick Actions</p>
              <div className="space-y-2">
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <Sparkles className="mr-2 h-4 w-4" />
                  Suggest Maintenance
                </Button>
                <Button variant="outline" className="w-full justify-start" size="sm">
                  <FileText className="mr-2 h-4 w-4" />
                  Generate Report
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
