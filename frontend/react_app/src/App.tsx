import React, { useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CheckIcon } from 'lucide-react'
import { TopBar } from './components/TopBar'
import { LayerRail } from './components/LayerRail'
import { MapCanvas } from './components/MapCanvas'
import { MapOverlays } from './components/MapOverlays'
import { InsightPanel } from './components/InsightPanel'
import { TimeSeriesStrip } from './components/TimeSeriesStrip'
import { ChatWidget } from './components/ChatWidget'
import type { LayerId } from './types/marine'

import { useTelemetry } from './hooks/useTelemetry'
import { useApi } from './hooks/useApi'

type Basemap = 'bathymetric' | 'satellite' | 'minimal'

interface AppProps {
  density?: 'comfortable' | 'compact'
  basemap?: Basemap
  showTimeSeries?: boolean
}

const DEFAULT_LAYERS: Record<LayerId, boolean> = {
  sst: true,
  chlorophyll: false,
  buoys: true,
  pfz: true,
  vessels: true,
  edna: true,
  anomalies: true,
}

export function App({
  density = 'comfortable',
  basemap = 'bathymetric',
  showTimeSeries = true,
}: AppProps) {
  const [activeLayers, setActiveLayers] = useState<Record<LayerId, boolean>>(DEFAULT_LAYERS)
  const [mapStyle, setMapStyle] = useState<Basemap>(basemap)
  const [selectedStationId, setSelectedStationId] = useState<string | null>(null)
  const [timeWindow, setTimeWindow] = useState('Last 48 h')
  const [exporting, setExporting] = useState(false)
  const [exported, setExported] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [flyTarget, setFlyTarget] = useState<[number, number] | null>(null)
  const timers = useRef<number[]>([])

  // Hooks for backend integration
  const { status: wsStatus, stations, alerts } = useTelemetry('ws://localhost:8000/ws/live-telemetry')
  const { pfzZones, species } = useApi('http://localhost:8000')

  const filteredStations = useMemo(() => {
    if (!searchQuery.trim()) return stations;
    const q = searchQuery.toLowerCase();
    return stations.filter(s => s.name?.toLowerCase().includes(q) || s.id?.toLowerCase().includes(q));
  }, [stations, searchQuery]);

  const filteredZones = useMemo(() => {
    if (!searchQuery.trim()) return pfzZones;
    const q = searchQuery.toLowerCase();
    return pfzZones.filter(z => 
      z.name?.toLowerCase().includes(q) || 
      z.targetSpecies?.toLowerCase().includes(q)
    );
  }, [pfzZones, searchQuery]);

  const filteredSpecies = useMemo(() => {
    if (!searchQuery.trim()) return species;
    const q = searchQuery.toLowerCase();
    return species.filter(s => 
      s.common?.toLowerCase().includes(q) || 
      s.scientific?.toLowerCase().includes(q) ||
      s.driver?.toLowerCase().includes(q)
    );
  }, [species, searchQuery]);

  const selectedStation = useMemo(
    () => stations.find((s) => s.id === selectedStationId) ?? null,
    [selectedStationId, stations],
  )
  const anomalyCount = useMemo(
    () => stations.filter((s) => s.status === 'anomaly').length,
    [stations],
  )

  const toggleLayer = (id: LayerId) =>
    setActiveLayers((prev) => ({ ...prev, [id]: !prev[id] }))

  const handleExport = () => {
    setExporting(true)
    timers.current.forEach(clearTimeout)
    timers.current = [
      window.setTimeout(() => {
        setExporting(false)
        setExported(true)
      }, 1400),
      window.setTimeout(() => setExported(false), 4200),
    ]
  }

  return (
    <div className="flex h-screen w-full flex-col bg-abyss-950 text-foam">
      <TopBar
        window={timeWindow}
        onWindowChange={setTimeWindow}
        onExport={handleExport}
        exporting={exporting}
        wsStatus={wsStatus}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
      />

      <div className="flex min-h-0 flex-1">
        <LayerRail
          activeLayers={activeLayers}
          onToggleLayer={toggleLayer}
          basemap={mapStyle}
          onBasemapChange={setMapStyle}
          dense={density === 'compact'}
          alerts={alerts}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            <MapCanvas
              basemap={mapStyle}
              activeLayers={activeLayers}
              stations={filteredStations}
              pfzZones={filteredZones}
              selectedStationId={selectedStationId}
              onSelectStation={setSelectedStationId}
              flyToCoords={flyTarget}
            />
            <MapOverlays
              activeLayers={activeLayers}
              anomalyCount={anomalyCount}
              hasSelection={Boolean(selectedStation)}
            />
            <AnimatePresence>
              {exported && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.98 }}
                  transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
                  role="status"
                  className="absolute bottom-4 left-1/2 z-[600] flex -translate-x-1/2 items-center gap-2 rounded-md border border-bio/40 bg-abyss-800 px-3 py-2 text-xs text-foam shadow-2xl"
                >
                  <CheckIcon className="h-3.5 w-3.5 text-bio" aria-hidden="true" />
                  Report compiled — 4 layers, {timeWindow.toLowerCase()}, 5 sources cited
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {showTimeSeries && <TimeSeriesStrip timeWindow={timeWindow} />}
        </main>

        <InsightPanel 
          station={selectedStation} 
          onClearStation={() => setSelectedStationId(null)}
          species={filteredSpecies}
        />
      </div>

      {/* Floating GraphRAG AI Chat Assistant */}
      <ChatWidget 
        onFlyTo={setFlyTarget} 
        onSearchSelect={setSearchQuery} 
      />
    </div>
  )
}
