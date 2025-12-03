import { useEffect, useState, useRef } from "react";
import { ShoppingBag, Package, Store, Star, ExternalLink, MessageCircle } from "lucide-react";

export default function App() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedImage, setExpandedImage] = useState(null);
  const [userId, setUserId] = useState(null);

  // zoom + pan state
  const imgRef = useRef(null);
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [origin, setOrigin] = useState({ x: 0, y: 0 });
  const [translation, setTranslation] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Get user ID from URL query params
    const params = new URLSearchParams(window.location.search);
    const userParam = params.get("user");
    
    if (!userParam) {
      setError("No user ID provided. Please use the link from WhatsApp.");
      setLoading(false);
      return;
    }

    setUserId(userParam);

    const fetchData = async () => {
      try {
        const res = await fetch(`http://localhost:5001/images/${userParam}`);
        
        if (!res.ok) {
          throw new Error(`Failed to fetch: ${res.status}`);
        }
        
        const data = await res.json();
        
        if (data.products && data.products.length > 0) {
          setProducts(data.products);
        } else {
          setError("No products found. Try searching on WhatsApp first!");
        }
      } catch (error) {
        console.error("Failed to fetch products:", error);
        setError("Could not load products. Make sure the backend is running on port 5001.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  // Freeze background scroll when viewer is open
  useEffect(() => {
    if (expandedImage) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [expandedImage]);

  const closeViewer = () => {
    setExpandedImage(null);
    setScale(1);
    setTranslation({ x: 0, y: 0 });
  };

  const startDrag = (e) => {
    e.preventDefault();
    setIsDragging(true);
    setOrigin({
      x: e.clientX - translation.x,
      y: e.clientY - translation.y,
    });
  };

  const duringDrag = (e) => {
    if (!isDragging) return;
    setTranslation({
      x: e.clientX - origin.x,
      y: e.clientY - origin.y,
    });
  };

  const stopDrag = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const newScale = Math.min(Math.max(1, scale + e.deltaY * -0.0015), 4);
    setScale(newScale);
  };

  const handleDoubleClick = () => {
    if (scale === 1) {
      setScale(2);
    } else {
      setScale(1);
      setTranslation({ x: 0, y: 0 });
    }
  };

  // Pinch-to-zoom for trackpads / touch
  useEffect(() => {
    if (!expandedImage) return;

    let pointers = new Map();
    let lastDistance = null;

    const container = containerRef.current;

    const getDistance = (p1, p2) =>
      Math.hypot(p1.x - p2.x, p1.y - p2.y);

    const pointerDown = (e) => {
      e.preventDefault();
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    };

    const pointerMove = (e) => {
      if (!pointers.has(e.pointerId)) return;
      e.preventDefault();

      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.size === 2) {
        const [p1, p2] = [...pointers.values()];
        const distance = getDistance(p1, p2);

        if (lastDistance !== null) {
          const diff = distance - lastDistance;
          const newScale = Math.min(Math.max(1, scale + diff * 0.005), 4);
          setScale(newScale);
        }
        lastDistance = distance;
      }
    };

    const pointerUp = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) lastDistance = null;
    };

    const preventBrowserZoom = (e) => {
      if (expandedImage) e.preventDefault();
    };

    container.addEventListener("pointerdown", pointerDown);
    container.addEventListener("pointermove", pointerMove);
    container.addEventListener("pointerup", pointerUp);
    container.addEventListener("pointercancel", pointerUp);
    container.addEventListener("wheel", preventBrowserZoom, { passive: false });
    document.addEventListener("gesturestart", preventBrowserZoom);
    document.addEventListener("gesturechange", preventBrowserZoom);

    return () => {
      container.removeEventListener("pointerdown", pointerDown);
      container.removeEventListener("pointermove", pointerMove);
      container.removeEventListener("pointerup", pointerUp);
      container.removeEventListener("pointercancel", pointerUp);
      container.removeEventListener("wheel", preventBrowserZoom);
      document.removeEventListener("gesturestart", preventBrowserZoom);
      document.removeEventListener("gesturechange", preventBrowserZoom);
    };
  }, [expandedImage, scale]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-purple-200 border-t-purple-600"></div>
          <p className="text-gray-600 font-medium">Loading your products...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
        <div className="text-center space-y-4 max-w-md">
          <div className="text-6xl">😕</div>
          <h2 className="text-2xl font-bold text-gray-800">{error}</h2>
          <p className="text-gray-600">Try searching for products on WhatsApp first, then click the link we send you.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12 space-y-3">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 bg-clip-text text-transparent">
            Your Search Results
          </h1>
          <p className="text-gray-600 text-lg">Found {products.length} amazing products for you</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {products.map((p, i) => (
            <div
              key={i}
              className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-100 hover:border-purple-200"
            >
              <div className="relative overflow-hidden bg-gradient-to-br from-gray-100 to-gray-50">
                <img
                  src={p.url}
                  alt={p.name}
                  onClick={() => setExpandedImage(p.url)}
                  className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-500 cursor-pointer"
                  onError={(e) => {
                    e.target.src = "https://via.placeholder.com/400x400?text=Image+Not+Available";
                  }}
                />
                <div className="absolute top-3 right-3 bg-white/95 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg">
                  <p className="text-lg font-bold text-purple-600">₹{p.price}</p>
                </div>
                {p.rating && p.rating !== "N/A" && (
                  <div className="absolute top-3 left-3 bg-amber-400/95 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1">
                    <Star className="w-4 h-4 fill-white text-white" />
                    <p className="text-sm font-bold text-white">{p.rating}</p>
                  </div>
                )}
              </div>

              <div className="p-5 space-y-4">
                <div className="space-y-2">
                  <h2 className="text-lg font-bold text-gray-900 leading-tight line-clamp-2 min-h-[3.5rem]">
                    {p.name}
                  </h2>
                  
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Store className="w-4 h-4 text-purple-600" />
                    <span className="font-medium">{p.store}</span>
                  </div>

                  {p.short_id && (
                    <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-medium">
                      <Package className="w-3 h-3" />
                      ID: {p.short_id}
                    </div>
                  )}
                </div>

                <div className="flex gap-2 pt-2 border-t border-gray-200">
                  <a
                    href={`https://wa.me/15551935302?text=${encodeURIComponent(
                      `Hi, I'm interested in ${p.name} (ID: ${p.short_id})`
                    )}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="
                      flex-1 flex items-center justify-center gap-2
                      text-center font-semibold text-white text-sm
                      bg-gradient-to-br from-green-600 to-emerald-700
                      hover:from-green-700 hover:to-emerald-800
                      py-2.5 rounded-lg
                      shadow-md hover:shadow-lg
                      transition-all duration-300
                      hover:-translate-y-0.5 active:scale-95
                    "
                  >
                    <MessageCircle className="w-4 h-4" />
                    Chat
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Fullscreen Zoom + Pan Viewer */}
      {expandedImage && (
        <div
          ref={containerRef}
          className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4 touch-none"
          onMouseMove={duringDrag}
          onMouseUp={stopDrag}
          onMouseLeave={stopDrag}
          onWheel={handleWheel}
          onClick={(e) => {
            if (e.target === containerRef.current) closeViewer();
          }}
        >
          <button
            onClick={closeViewer}
            className="absolute top-4 right-4 text-white bg-black/50 hover:bg-black/70 rounded-full p-3 transition-colors z-10"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-full text-sm">
            Zoom: {Math.round(scale * 100)}% • Double-click to reset
          </div>
          
          <img
            ref={imgRef}
            src={expandedImage}
            alt="Expanded"
            onMouseDown={startDrag}
            onDoubleClick={handleDoubleClick}
            draggable={false}
            className="max-w-none max-h-none rounded-xl shadow-2xl select-none cursor-grab active:cursor-grabbing transition-all"
            style={{
              transform: `translate(${translation.x}px, ${translation.y}px) scale(${scale})`,
              maxWidth: "90%",
              maxHeight: "90%",
            }}
          />
        </div>
      )}
    </div>
  );
}