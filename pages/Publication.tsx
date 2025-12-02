import React from 'react';
import { ExternalLink, Github, FileText } from 'lucide-react';
import { PUBLICATIONS } from '../constants';

const Publication: React.FC = () => {
  return (
    <div className="animate-fade-in pt-12 pb-20">
       <div className="max-w-4xl mx-auto">
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between mb-12">
          <h1 className="text-4xl md:text-5xl font-serif font-light text-anthropic-text mb-4 md:mb-0">Publications</h1>
          <p className="text-anthropic-gray font-sans font-light">
            Selected papers & Preprints
          </p>
        </div>

        <div className="space-y-8">
           {PUBLICATIONS.map((pub) => {
             // Check if it's the "Reasoning to Align" paper for special styling
             const isReasoningToAlign = pub.tags?.includes("Reasoning to Align");
             const containerClasses = isReasoningToAlign
               ? "bg-anthropic-leaf/10 border-anthropic-leaf/40 hover:border-anthropic-leaf"
               : "border-anthropic-stone hover:border-anthropic-accent";
             
             return (
               <div key={pub.id} className={`group border-l-4 pl-6 md:pl-8 py-4 pr-4 rounded-r-lg transition-colors duration-300 ${containerClasses}`}>
                 <div className="flex flex-wrap gap-2 mb-3">
                   {pub.preprint && (
                     <span className="px-2 py-0.5 bg-anthropic-stone/50 rounded text-xs font-medium text-anthropic-text uppercase tracking-wide">
                       Preprint
                     </span>
                   )}
                   {pub.tags?.map(tag => (
                     <span 
                       key={tag} 
                       className={`px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wide ${
                         tag === "Reasoning to Align" 
                           ? "bg-anthropic-leaf/30 text-anthropic-text" 
                           : "bg-anthropic-leaf/20 text-anthropic-text"
                       }`}
                     >
                       {tag}
                     </span>
                   ))}
                 </div>
                 
                 <h3 className="text-xl md:text-2xl font-serif text-anthropic-text mb-3 leading-tight">
                   {pub.link ? (
                     <a href={pub.link} target="_blank" rel="noreferrer" className="hover:text-anthropic-accent transition-colors">
                       {pub.title}
                     </a>
                   ) : (
                     pub.title
                   )}
                 </h3>
                 
                 <p className="text-anthropic-gray text-base mb-3 font-light">
                   {pub.authors.map((author, idx) => (
                     <span key={idx} className={author.includes("Rui Wu") ? "font-bold text-anthropic-text" : ""}>
                       {author}{idx < pub.authors.length - 1 ? ", " : ""}
                     </span>
                   ))}
                 </p>

                 <div className="text-sm text-anthropic-gray/80 italic mb-4">
                   {pub.venue} ({pub.year})
                 </div>

                 <div className="flex gap-4">
                   {pub.link && (
                     <a 
                       href={pub.link} 
                       target="_blank" 
                       rel="noreferrer"
                       className="flex items-center text-sm font-medium text-anthropic-accent hover:text-anthropic-text transition-colors"
                     >
                       <FileText size={16} className="mr-1.5" />
                       ArXiv
                     </a>
                   )}
                   {pub.github && (
                     <a 
                       href={pub.github} 
                       target="_blank" 
                       rel="noreferrer"
                       className="flex items-center text-sm font-medium text-anthropic-accent hover:text-anthropic-text transition-colors"
                     >
                       <Github size={16} className="mr-1.5" />
                       Code
                     </a>
                   )}
                 </div>
               </div>
             );
           })}
        </div>
      </div>
    </div>
  );
};

export default Publication;