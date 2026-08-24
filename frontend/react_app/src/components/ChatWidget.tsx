import React, { useState, useRef, useEffect } from 'react'
import { MessageSquareIcon, XIcon, SendIcon, SparklesIcon, BotIcon, UserIcon, MapPinIcon } from 'lucide-react'

interface Message {
  id: string
  sender: 'user' | 'bot'
  text: string
  sources?: string[]
  coords?: [number, number]
  zoneId?: string
}

interface ChatWidgetProps {
  onFlyTo?: (coords: [number, number]) => void
  onSearchSelect?: (term: string) => void
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: 'welcome-1',
    sender: 'bot',
    text: "👋 **I am BlueByte AI.**\n\nI am connected to live ocean telemetry, active PFZ zones, and the GNN biodiversity knowledge graph. Ask me anything about fishing hotspots, marine heatwaves, or species migration!",
    sources: ["BlueByte GraphRAG Engine", "INCOIS Telemetry"]
  }
]

const QUICK_PROMPTS = [
  "Where are Sardines near Goa?",
  "Show active heatwave alerts",
  "Yellowfin Tuna in Lakshadweep",
  "How does GNN predict species?"
]

export function ChatWidget({ onFlyTo, onSearchSelect }: ChatWidgetProps) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, open])

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || input.trim()
    if (!textToSend || loading) return

    const userMsg: Message = {
      id: `usr-${Date.now()}`,
      sender: 'user',
      text: textToSend
    }

    setMessages(prev => [...prev, userMsg])
    if (!queryText) setInput('')
    setLoading(true)

    try {
      // Send to FastAPI backend
      const res = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend })
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const botMsg: Message = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        text: data.reply,
        sources: data.sources || ["Knowledge Graph", "INCOIS Telemetry"],
        coords: data.target_coords ? [data.target_coords[0], data.target_coords[1]] : undefined,
        zoneId: data.highlight_zone
      }

      setMessages(prev => [...prev, botMsg])

      // Auto fly map if coordinates returned
      if (data.target_coords && onFlyTo) {
        onFlyTo([data.target_coords[0], data.target_coords[1]])
      }
    } catch (err) {
      // Fallback local smart response if FastAPI backend is offline
      const fallbackReply = generateClientFallback(textToSend)
      const botMsg: Message = {
        id: `bot-${Date.now()}`,
        sender: 'bot',
        text: fallbackReply.text,
        sources: ["Local Knowledge Graph (Offline Mode)", "GNN Rules"],
        coords: fallbackReply.coords,
        zoneId: fallbackReply.zoneId
      }
      setMessages(prev => [...prev, botMsg])

      if (fallbackReply.coords && onFlyTo) {
        onFlyTo(fallbackReply.coords)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed bottom-20 right-6 z-[1000] font-sans">
      {/* Floating Trigger Button */}
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="group flex items-center gap-2 rounded-full border border-tide/50 bg-abyss-900/90 px-4 py-2.5 text-xs font-semibold text-foam shadow-[0_8px_32px_rgba(0,229,255,0.25)] backdrop-blur-md transition-all duration-200 hover:scale-105 hover:border-tide hover:bg-abyss-800"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-tide opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-tide" />
          </span>
          <SparklesIcon className="h-4 w-4 text-tide" />
          <span>Ask Marine AI</span>
          <span className="rounded bg-abyss-700 px-1.5 py-0.5 font-mono text-[10px] text-tide">GraphRAG</span>
        </button>
      )}

      {/* Expanded Chat Window */}
      {open && (
        <div className="flex h-[520px] w-[380px] flex-col overflow-hidden rounded-xl border border-line bg-abyss-900/95 shadow-[0_16px_48px_rgba(0,0,0,0.8)] backdrop-blur-2xl transition-all">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-line bg-abyss-950/60 px-4 py-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-tide/30 bg-tide/10">
                <BotIcon className="h-4 w-4 text-tide" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-semibold text-foam">BlueByte Marine Assistant</h3>
                  <span className="rounded border border-bio/30 bg-bio/10 px-1 py-0.2 font-mono text-[9px] text-bio">
                    GraphRAG
                  </span>
                </div>
                <p className="text-[10px] text-foam-dim">Ground-truth oceanography & GNN intelligence</p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-md p-1 text-foam-dim hover:bg-abyss-800 hover:text-foam"
            >
              <XIcon className="h-4 w-4" />
            </button>
          </div>

          {/* Quick Prompts Bar */}
          <div className="flex gap-1.5 overflow-x-auto border-b border-line/50 bg-abyss-950/30 px-3 py-2 scrollbar-none">
            {QUICK_PROMPTS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSend(p)}
                className="shrink-0 rounded-full border border-line bg-abyss-800/80 px-2.5 py-1 text-[10px] text-foam-muted transition-colors hover:border-tide/50 hover:text-tide"
              >
                {p}
              </button>
            ))}
          </div>

          {/* Messages Feed */}
          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-3 text-xs">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-2 ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {m.sender === 'bot' && (
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-tide/30 bg-tide/10">
                    <BotIcon className="h-3.5 w-3.5 text-tide" />
                  </div>
                )}
                <div
                  className={`max-w-[85%] rounded-lg px-3 py-2 leading-relaxed ${
                    m.sender === 'user'
                      ? 'bg-tide text-abyss-950 font-medium'
                      : 'border border-line bg-abyss-800 text-foam'
                  }`}
                >
                  <div className="whitespace-pre-wrap text-[11px]">{m.text}</div>
                  
                  {/* Action Link for coordinates */}
                  {m.coords && (
                    <button
                      type="button"
                      onClick={() => onFlyTo && onFlyTo(m.coords!)}
                      className="mt-2 flex items-center gap-1 rounded bg-abyss-950/60 px-2 py-1 font-mono text-[10px] text-tide hover:underline"
                    >
                      <MapPinIcon className="h-3 w-3" />
                      <span>Focus map at [{m.coords[0]}°N, {m.coords[1]}°E]</span>
                    </button>
                  )}

                  {/* Sources tag */}
                  {m.sources && m.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1 border-t border-line/40 pt-1.5 text-[9px] text-foam-dim">
                      <span>Sources:</span>
                      {m.sources.map((s, i) => (
                        <span key={i} className="rounded bg-abyss-900 px-1 py-0.5 text-foam-muted">
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {m.sender === 'user' && (
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-tide/20 text-tide">
                    <UserIcon className="h-3.5 w-3.5" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-[11px] text-foam-muted">
                <div className="h-2 w-2 animate-bounce rounded-full bg-tide" />
                <div className="h-2 w-2 animate-bounce rounded-full bg-tide [animation-delay:0.2s]" />
                <div className="h-2 w-2 animate-bounce rounded-full bg-tide [animation-delay:0.4s]" />
                <span>Querying GraphRAG knowledge base…</span>
              </div>
            )}
          </div>

          {/* Input Footer */}
          <div className="border-t border-line bg-abyss-950/60 p-2.5">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSend()
              }}
              className="flex items-center gap-2"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about fish, zones, heatwaves…"
                className="h-8 flex-1 rounded-md border border-line bg-abyss-800 px-3 text-xs text-foam placeholder:text-foam-dim focus:border-tide focus:outline-none focus:ring-1 focus:ring-tide/40"
              />
              <button
                type="submit"
                disabled={!input.trim() || loading}
                className="flex h-8 w-8 items-center justify-center rounded-md bg-tide text-abyss-950 transition-opacity hover:bg-tide/90 disabled:opacity-50"
              >
                <SendIcon className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

function generateClientFallback(query: string): { text: string; coords?: [number, number]; zoneId?: string } {
  const q = query.toLowerCase()
  if (q.includes('sardine') || q.includes('goa') || q.includes('karwar')) {
    return {
      text: "🐟 **Indian Oil Sardine Advisory**\n\n• **Location**: Malpe–Karwar Upwelling Front (`13.9°N, 73.4°E`)\n• **Confidence**: **91%** GNN habitat match\n• **Driver**: Coastal upwelling front with SST ~28.8°C and high chlorophyll (2.4 mg/m³).\n• **Recommendation**: Prime conditions for purse-seine vessels off Karnataka/Goa.",
      coords: [13.9, 73.4],
      zoneId: 'PFZ-AS-04'
    }
  }
  if (q.includes('tuna') || q.includes('lakshadweep')) {
    return {
      text: "🦈 **Yellowfin Tuna Advisory**\n\n• **Location**: Lakshadweep Thermal Ridge (`11.4°N, 71.2°E`)\n• **Confidence**: **78%** GNN prediction\n• **Driver**: Deep thermal ridge with SST ~30.6°C.\n• **Recommendation**: Ideal for longline operations in deep pelagic shelf.",
      coords: [11.4, 71.2],
      zoneId: 'PFZ-AS-11'
    }
  }
  if (q.includes('mackerel') || q.includes('godavari')) {
    return {
      text: "🐟 **Indian Mackerel Advisory**\n\n• **Location**: Godavari Plume Convergence (`16.6°N, 82.1°E`)\n• **Confidence**: **66%** probability\n• **Driver**: Nutrient-rich river plume mixing front in Bay of Bengal.",
      coords: [16.6, 82.1],
      zoneId: 'PFZ-BB-06'
    }
  }
  if (q.includes('alert') || q.includes('heatwave') || q.includes('anomaly')) {
    return {
      text: "⚠️ **Active Marine Anomalies**\n\n• **Marine Heatwave**: Sensor `BD08` in Central Arabian Sea recorded **29.8°C** (+3.4 Z-score).\n• **Low Oxygen Zone**: Station `CM03` reported dissolved oxygen at **3.2 mg/L**.\n• Pelagic schools may dive deeper to seek thermal refuge.",
      coords: [15.42, 69.24]
    }
  }
  if (q.includes('gnn') || q.includes('edna') || q.includes('model') || q.includes('graph')) {
    return {
      text: "🧠 **Graph Neural Network (GNN) Engine**\n\n• Heterogeneous Graph linking `OceanGrids ↔ Species ↔ eDNA sequences`.\n• Uses GAT message-passing to predict unseen fish occurrence from environmental vectors (SST, Salinity, Chlorophyll) and genetic barcodes (COI/12S/16S).",
    }
  }
  return {
    text: "🌊 **BlueByte Marine AI Ready**\n\nI can analyze live INCOIS telemetry, PFZ fishing zones, and GNN biodiversity predictions.\n\nTry asking:\n• *'Where are Sardines near Goa?'*\n• *'Show active heatwave alerts'*\n• *'Yellowfin Tuna in Lakshadweep'*"
  }
}
