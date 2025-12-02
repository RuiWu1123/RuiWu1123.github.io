
import React, { useState } from 'react';
import WorldMap from '../components/WorldMap';
import { VisitedPlace } from '../types';
import { MapPin, Calendar, Camera, Hash } from 'lucide-react';

const Travel: React.FC = () => {
  const [selectedPlace, setSelectedPlace] = useState<VisitedPlace | null>(null);

  return (
    <div className="animate-fade-in pt-12 pb-20">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-serif font-light mb-4 text-anthropic-text">Travel Gallery</h1>
          <p className="text-lg text-anthropic-gray font-sans font-light max-w-2xl mx-auto">
            A record of my journey around the globe. Select a marked location to view details.
          </p>
        </div>

        {/* Map Container */}
        <div className="mb-12 shadow-sm">
          <WorldMap 
            selectedPlaceId={selectedPlace?.id || null} 
            onPlaceSelect={setSelectedPlace} 
          />
          <div className="flex justify-center mt-4 gap-6 text-sm text-anthropic-gray">
            <div className="flex items-center gap-2">
               <span className="w-3 h-3 rounded-full bg-anthropic-accent"></span> Visited
            </div>
            <div className="flex items-center gap-2">
               <span className="w-3 h-3 rounded-full bg-[#191919]"></span> Selected
            </div>
          </div>
        </div>

        {/* Details Section - Only visible when selected */}
        {selectedPlace ? (
          <div className="bg-white border border-anthropic-text/5 rounded-xl overflow-hidden shadow-sm transition-all duration-500 ease-in-out animate-fade-in">
            {/* Header Info */}
            <div className="p-8 border-b border-anthropic-text/5 flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div>
                <div className="flex items-center gap-2 text-anthropic-accent mb-2">
                  <MapPin size={18} />
                  <span className="text-sm font-semibold tracking-wide uppercase">{selectedPlace.name}</span>
                </div>
                <h2 className="text-3xl font-serif text-anthropic-text mb-2">{selectedPlace.name}</h2>
                <div className="flex items-center gap-4 text-anthropic-gray text-sm">
                  <div className="flex items-center gap-1.5">
                    <Calendar size={16} />
                    <span>{selectedPlace.date}</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-anthropic-stone/50 px-2 py-0.5 rounded-full text-anthropic-text font-medium">
                    <Hash size={14} />
                    <span>Visited {selectedPlace.visitCount} {selectedPlace.visitCount === 1 ? 'time' : 'times'}</span>
                  </div>
                </div>
              </div>
              <div className="max-w-xl text-anthropic-gray font-light leading-relaxed">
                {selectedPlace.description}
              </div>
            </div>

            {/* Horizontal Scroll Gallery */}
            <div className="bg-anthropic-bg/50 p-8">
              <div className="flex items-center gap-2 mb-4 text-anthropic-gray/60 text-sm font-sans uppercase tracking-widest">
                <Camera size={14} />
                <span>Gallery ({selectedPlace.images.length} photos)</span>
              </div>
              
              {/* Scroll Container */}
              <div className="flex overflow-x-auto gap-6 pb-6 scrollbar-thin scrollbar-thumb-anthropic-stone scrollbar-track-transparent snap-x snap-mandatory">
                {selectedPlace.images.map((imgUrl, index) => (
                  <div 
                    key={index} 
                    className="flex-none snap-center relative group rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow"
                  >
                    {/* Fixed height for vertical alignment, width auto preserves aspect ratio */}
                    <img 
                      src={imgUrl} 
                      alt={`${selectedPlace.name} scene ${index + 1}`} 
                      className="h-80 md:h-96 w-auto object-cover transition-transform duration-700 group-hover:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-300" />
                  </div>
                ))}
                
                {/* Spacer for end of scroll */}
                <div className="w-1 flex-none" />
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-16 bg-anthropic-stone/20 rounded-xl border border-dashed border-anthropic-text/10">
             <p className="text-anthropic-gray font-light text-lg">Select a marker on the map to view my journey.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Travel;
