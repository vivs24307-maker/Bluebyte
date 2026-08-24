import React, { useEffect, useRef } from 'react'
import L from 'leaflet'
import type { LayerId, Station } from '../types/marine'
import { ednaSites, vesselTracks } from '../data/zones'
import { oceanField } from '../utils/field'
import { sstColor } from '../utils/series'

const BASEMAPS: Record<string, string> = {
  bathymetric: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  minimal: 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  satellite:
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
}

interface MapCanvasProps {
  basemap: 'bathymetric' | 'satellite' | 'minimal'
  activeLayers: Record<LayerId, boolean>
  stations: Station[]
  pfzZones: any[]
  selectedStationId: string | null
  onSelectStation: (id: string) => void
  flyToCoords?: [number, number] | null
}

export function MapCanvas({
  basemap,
  activeLayers,
  stations,
  pfzZones,
  selectedStationId,
  onSelectStation,
  flyToCoords,
}: MapCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<L.Map | null>(null)
  const tileRef = useRef<L.TileLayer | null>(null)
  const overlayRef = useRef<L.LayerGroup | null>(null)
  const selectRef = useRef(onSelectStation)
  selectRef.current = onSelectStation

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: false,
      minZoom: 4,
      maxZoom: 9,
      zoomSnap: 0.5,
    }).setView([14.6, 79.5], 5)
    L.control.zoom({ position: 'bottomright' }).addTo(map)
    mapRef.current = map
    overlayRef.current = L.layerGroup().addTo(map)
    return () => {
      map.remove()
      mapRef.current = null
      overlayRef.current = null
      tileRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (tileRef.current) map.removeLayer(tileRef.current)
    const tile = L.tileLayer(BASEMAPS[basemap], {
      opacity: basemap === 'satellite' ? 0.75 : 1,
    })
    tile.addTo(map)
    tile.getContainer()?.style.setProperty('filter', 'saturate(0.7) brightness(0.85)')
    tileRef.current = tile
  }, [basemap])

  useEffect(() => {
    const group = overlayRef.current
    if (!group) return
    group.clearLayers()

    if (activeLayers.sst || activeLayers.chlorophyll) {
      oceanField.forEach((cell) => {
        if (activeLayers.sst) {
          L.circleMarker([cell.lat, cell.lng], {
            radius: 9,
            stroke: false,
            fillColor: sstColor(cell.sst),
            fillOpacity: 0.26,
            interactive: false,
          }).addTo(group)
        }
        if (activeLayers.chlorophyll && cell.chlorophyll > 0.9) {
          L.circleMarker([cell.lat, cell.lng], {
            radius: 4 + cell.chlorophyll * 2.4,
            stroke: false,
            fillColor: '#56d9a3',
            fillOpacity: 0.2,
            interactive: false,
          }).addTo(group)
        }
      })
    }

    if (activeLayers.pfz) {
      pfzZones.forEach((zone) => {
        L.polygon(zone.coords, {
          color: '#f2a03d',
          weight: 1.5,
          fillColor: '#f2a03d',
          fillOpacity: 0.1 + zone.confidence * 0.14,
          dashArray: zone.confidence < 0.75 ? '4 4' : undefined,
        })
          .bindTooltip(
            `<strong>${zone.name}</strong><br/>PFZ ${zone.id} · ${Math.round(
              zone.confidence * 100,
            )}% confidence<br/>Target: ${zone.targetSpecies}`,
            { className: 'bb-tooltip', sticky: true },
          )
          .addTo(group)
      })
    }

    if (activeLayers.vessels) {
      vesselTracks.forEach((track) => {
        const color =
          track.risk === 'flagged' ? '#f2624a' : track.risk === 'review' ? '#f2a03d' : '#8fadbe'
        L.polyline(track.path, {
          color,
          weight: track.risk === 'clear' ? 1.2 : 2,
          opacity: 0.9,
          dashArray: track.risk === 'review' ? '3 5' : undefined,
        })
          .bindTooltip(`<strong>${track.name}</strong><br/>${track.id} · ${track.risk}`, {
            className: 'bb-tooltip',
            sticky: true,
          })
          .addTo(group)
        const head = track.path[track.path.length - 1]
        L.circleMarker(head, { radius: 3, color, weight: 1, fillOpacity: 1 }).addTo(group)
      })
    }

    if (activeLayers.edna) {
      ednaSites.forEach((site) => {
        L.circleMarker([site.lat, site.lng], {
          radius: 5 + site.novelTaxa * 0.35,
          color: '#a78bfa',
          weight: 1.5,
          fillColor: '#a78bfa',
          fillOpacity: 0.35,
        })
          .bindTooltip(
            `<strong>${site.name}</strong><br/>${site.id} · ${site.richness} taxa<br/>${site.novelTaxa} unassigned · ${site.sampledDaysAgo} d ago`,
            { className: 'bb-tooltip', sticky: true },
          )
          .addTo(group)
      })
    }

    if (activeLayers.buoys) {
      stations.forEach((station) => {
        const selected = station.id === selectedStationId
        const color =
          station.status === 'anomaly'
            ? '#f2624a'
            : station.status === 'offline'
              ? '#5f8298'
              : '#37c8e0'
        L.circleMarker([station.lat, station.lng], {
          radius: selected ? 8 : 5.5,
          color: selected ? '#ffffff' : color,
          weight: selected ? 2 : 1.5,
          fillColor: color,
          fillOpacity: station.status === 'offline' ? 0.2 : 0.85,
        })
          .bindTooltip(
            `<strong>${station.name}</strong><br/>${station.agency} · ${station.sst} °C · ${station.oxygen} mg/L`,
            { className: 'bb-tooltip', sticky: true },
          )
          .on('click', () => selectRef.current(station.id))
          .addTo(group)
      })
    }

    if (activeLayers.anomalies) {
      stations
        .filter((s) => s.status === 'anomaly')
        .forEach((station) => {
          L.circleMarker([station.lat, station.lng], {
            radius: 16,
            color: '#f2624a',
            weight: 1,
            opacity: 0.6,
            fill: false,
            interactive: false,
          }).addTo(group)
        })
    }
  }, [activeLayers, stations, selectedStationId, pfzZones])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (flyToCoords) {
      map.flyTo(flyToCoords, 7.5, { duration: 1.2 })
    } else if (pfzZones.length === 1 && pfzZones[0].coords && pfzZones[0].coords.length > 0) {
      const bounds = L.latLngBounds(pfzZones[0].coords)
      map.flyToBounds(bounds, { padding: [100, 100], maxZoom: 8, duration: 1.2 })
    } else if (stations.length === 1) {
      map.flyTo([stations[0].lat, stations[0].lng], 7, { duration: 1.2 })
    }
  }, [pfzZones, stations, flyToCoords])

  return <div ref={containerRef} className="h-full w-full" role="application" aria-label="Marine data map" />
}
