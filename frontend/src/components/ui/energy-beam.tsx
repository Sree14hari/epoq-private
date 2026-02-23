import React, { useEffect, useRef } from 'react';

interface EnergyBeamProps {
    projectId?: string;
    className?: string;
}

declare global {
    interface Window {
        UnicornStudio?: any;
    }
}

const EnergyBeam: React.FC<EnergyBeamProps> = ({
    projectId = "hRFfUymDGOHwtFe7evR2",
    className = ""
}) => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let sceneInstance: any = null;
        let isMounted = true;
        let scriptElement: HTMLScriptElement | null = null;
        let loadHandler: (() => void) | null = null;

        const initScene = () => {
            if (window.UnicornStudio && containerRef.current) {
                // Initialize the Unicorn Studio project
                // The library automatically picks up data-us-* attributes
                window.UnicornStudio.init().then((scenes: any[]) => {
                    if (isMounted && scenes && scenes.length > 0) {
                        // Find the scene associated with THIS container
                        sceneInstance = scenes.find((s: any) => s.element === containerRef.current) || scenes[0];
                        console.log('Unicorn Studio initialized');
                    } else if (!isMounted && scenes) {
                        // If unmounted during init, destroy immediately
                        scenes.forEach((s: any) => s.destroy());
                    }
                }).catch((err: any) => {
                    console.error('Error initializing Unicorn Studio:', err);
                });
            }
        };

        const loadScript = () => {
            if (window.UnicornStudio) {
                initScene();
                return;
            }

            const scriptUrl = 'https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v1.5.2/dist/unicornStudio.umd.js';
            let script = document.querySelector(`script[src="${scriptUrl}"]`) as HTMLScriptElement;

            if (!script) {
                script = document.createElement('script');
                script.src = scriptUrl;
                script.async = true;
                document.head.appendChild(script);
            }

            scriptElement = script;
            loadHandler = () => {
                if (isMounted) initScene();
            };

            script.addEventListener('load', loadHandler);
        };

        loadScript();

        return () => {
            isMounted = false;

            if (scriptElement && loadHandler) {
                scriptElement.removeEventListener('load', loadHandler);
            }

            if (sceneInstance && typeof sceneInstance.destroy === 'function') {
                sceneInstance.destroy();
            }
        };
    }, [projectId]);

    return (
        <div className={`relative w-full h-screen bg-black overflow-hidden ${className}`}>
            <div
                ref={containerRef}
                data-us-project={projectId}
                data-us-scale="1" 
                data-us-fps="60"    
                data-us-dpi="0.9"   
                className="w-full h-full"
            />
        </div>
    );
};

export default EnergyBeam;
