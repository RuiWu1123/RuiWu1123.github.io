import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { VISITED_PLACES } from '../constants';
import { VisitedPlace } from '../types';

interface WorldMapProps {
  onPlaceSelect: (place: VisitedPlace | null) => void;
  selectedPlaceId: string | null;
}

const WorldMap: React.FC<WorldMapProps> = ({ onPlaceSelect, selectedPlaceId }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [geoJson, setGeoJson] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Load GeoJSON data once
  useEffect(() => {
    // A simplified GeoJSON of the world
    fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch");
        return response.json();
      })
      .then((data) => {
        setGeoJson(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading map data", err);
        setError(true);
        setLoading(false);
      });
  }, []);

  // Render Map
  useEffect(() => {
    if (!geoJson || !svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const width = containerRef.current.clientWidth;
    const height = 500; // Fixed height for map area

    svg.attr("width", width).attr("height", height);

    // Projection
    const projection = d3.geoMercator()
      .scale(width / 6.5)
      .translate([width / 2, height / 1.5]);

    const path = d3.geoPath().projection(projection);

    // Draw Countries
    svg.append("g")
      .selectAll("path")
      .data(geoJson.features)
      .join("path")
      .attr("d", path as any)
      .attr("fill", "#E6E4DD") // Anthropic stone/beige
      .attr("stroke", "#F4F3EF") // Background color for borders
      .attr("stroke-width", 0.5)
      .style("opacity", 0.8);

    // Draw Markers (Visited Places)
    const markers = svg.append("g")
      .selectAll("g")
      .data(VISITED_PLACES)
      .join("g")
      .attr("transform", d => {
        const coords = projection(d.coordinates);
        return coords ? `translate(${coords[0]}, ${coords[1]})` : null;
      })
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onPlaceSelect(d);
      });

    // Outer circle (pulse effect)
    markers.append("circle")
      .attr("r", 6)
      .attr("fill", "rgba(217, 119, 87, 0.3)") // Accent color transparent
      .attr("class", "animate-ping");

    // Inner circle
    markers.append("circle")
      .attr("r", 4)
      .attr("fill", (d) => d.id === selectedPlaceId ? "#191919" : "#D97757") // Black if selected, accent if not
      .attr("stroke", "#F4F3EF")
      .attr("stroke-width", 1)
      .transition()
      .duration(300)
      .attr("r", (d) => d.id === selectedPlaceId ? 6 : 4);

    // Click on background deselects
    svg.on("click", () => {
      onPlaceSelect(null);
    });

  }, [geoJson, selectedPlaceId, onPlaceSelect]);

  // Handle window resize logic broadly via effect dependency or layout
  useEffect(() => {
    const handleResize = () => {
        // Simple re-render trigger could go here, but D3 ref approach usually needs a full re-draw.
        // For this demo, we assume the container width is stable or a refresh is acceptable.
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (error) {
    return (
      <div className="w-full h-[500px] bg-[#EAE8E2]/30 rounded-lg flex items-center justify-center border border-anthropic-text/5 text-anthropic-gray">
        Map data unavailable. Please refresh or try again later.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full bg-[#EAE8E2]/30 rounded-lg overflow-hidden border border-anthropic-text/5 relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#EAE8E2]/50">
          <span className="text-anthropic-gray text-sm tracking-widest uppercase">Loading Map...</span>
        </div>
      )}
      <svg ref={svgRef} className="w-full h-[500px]"></svg>
    </div>
  );
};

export default WorldMap;