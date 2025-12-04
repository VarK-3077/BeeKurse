import { useEffect, useState, useRef } from "react";
import { ShoppingBag, Package, Store, Star, ExternalLink, ShoppingCart, ChevronDown, ChevronUp } from "lucide-react";

export default function App() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedImage, setExpandedImage] = useState(null);
  const [userId, setUserId] = useState(null);
  const [expandedDescriptions, setExpandedDescriptions] = useState({});

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
        // Use relative URL so it works both locally and via ngrok
        // The unified gateway proxies /images/{user} to Strontium API
        const res = await fetch(`/images/${userParam}`);
        
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

  const toggleDescription = (index) => {
    setExpandedDescriptions(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

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

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {products.map((p, i) => (
            <div
              key={i}
              className="group bg-white rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden border-2 border-gray-100 hover:border-purple-300 flex flex-col"
            >
              <div className="relative overflow-hidden bg-gradient-to-br from-gray-100 to-gray-50 h-80">
                <img
                  src={p.url || p.imageid}
                  alt={p.prod_name || p.name}
                  onClick={() => setExpandedImage(p.url || p.imageid)}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 cursor-pointer"
                  onError={(e) => {
                    e.target.src = "https://via.placeholder.com/400x400?text=Image+Not+Available";
                  }}
                />
                <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm px-4 py-2 rounded-full shadow-xl">
                  <p className="text-xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">₹{p.price}</p>
                </div>
                {p.rating && p.rating !== "N/A" && (
                  <div className="absolute top-4 left-4 bg-gradient-to-r from-amber-400 to-orange-500 px-3 py-2 rounded-full shadow-xl flex items-center gap-1.5">
                    <Star className="w-5 h-5 fill-white text-white" />
                    <p className="text-sm font-bold text-white">{p.rating}</p>
                  </div>
                )}
              </div>

              <div className="p-6 space-y-4 flex-1 flex flex-col">
                <div className="space-y-4 flex-1">
                  <h2 className="text-xl font-bold text-gray-900 leading-tight line-clamp-2">
                    {p.prod_name || p.name}
                  </h2>
                  
                  {p.brand && (
                    <div className="inline-flex items-center px-3 py-1.5 rounded-full bg-gradient-to-r from-purple-100 to-pink-100 border border-purple-200">
                      <p className="text-sm font-semibold text-purple-700">{p.brand}</p>
                    </div>
                  )}

                  <div className="flex items-center gap-2 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
                    <Store className="w-5 h-5 text-purple-600 flex-shrink-0" />
                    <div className="flex-1">
                      <span className="font-semibold block">{p.store}</span>
                      {p.store_contact && (
                        <span className="text-xs text-gray-500">📞 {p.store_contact}</span>
                      )}
                    </div>
                  </div>

                  {p.description && (
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
                      <p className={`text-sm text-gray-700 leading-relaxed ${expandedDescriptions[i] ? '' : 'line-clamp-3'}`}>
                        {p.description}
                      </p>
                      {p.description.length > 150 && (
                        <button
                          onClick={() => toggleDescription(i)}
                          className="mt-2 text-xs font-semibold text-purple-600 hover:text-purple-700 flex items-center gap-1 transition-colors"
                        >
                          {expandedDescriptions[i] ? (
                            <>Show less <ChevronUp className="w-3 h-3" /></>
                          ) : (
                            <>Read more <ChevronDown className="w-3 h-3" /></>
                          )}
                        </button>
                      )}
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    {p.colour && (
                      <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-gradient-to-r from-purple-100 to-pink-100 border border-purple-200 text-xs font-semibold text-purple-700">
                        🎨 {p.colour}
                      </span>
                    )}
                    
                    {p.stock !== undefined && (
                      <span className={`inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold border ${
                        p.stock > 10 
                          ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200 text-green-700' 
                          : p.stock > 0 
                          ? 'bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200 text-amber-700'
                          : 'bg-gradient-to-r from-red-50 to-rose-50 border-red-200 text-red-700'
                      }`}>
                        {p.stock > 0 ? `✓ ${p.stock} in stock` : '✗ Out of stock'}
                      </span>
                    )}

                    {p.quantity && p.quantityunit && (
                      <span className="inline-flex items-center px-3 py-1.5 rounded-full bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 text-xs font-semibold text-blue-700">
                        📦 {p.quantity} {p.quantityunit}
                      </span>
                    )}
                  </div>

                  {(p.size || p.dimensions) && (
                    <div className="text-xs text-gray-600 bg-gradient-to-r from-gray-50 to-slate-50 rounded-lg p-3 border border-gray-200">
                      <span className="font-semibold text-gray-700">📏 Dimensions: </span>
                      {p.size || `${p.dimensions.length} × ${p.dimensions.width} × ${p.dimensions.height} ${p.dimensions.unit}`}
                      {p.dimensions?.weight && (
                        <span className="block mt-1">
                          <span className="font-semibold text-gray-700">⚖️ Weight: </span>
                          {p.dimensions.weight}{p.dimensions.weight_unit}
                        </span>
                      )}
                    </div>
                  )}

                  {p.other_properties?.warranty_years && (
                    <div className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200">
                      <span className="text-lg">🛡️</span>
                      <span className="text-sm font-semibold text-green-700">
                        {p.other_properties.warranty_years} year warranty
                      </span>
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t-2 border-gray-100">
                  <a
                    href={`https://wa.me/15551935302?text=${encodeURIComponent(
                      `add ${p.short_id} to cart`
                    )}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="
                      w-full flex items-center justify-center gap-2
                      text-center font-bold text-white text-base
                      bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-700
                      hover:from-purple-700 hover:via-indigo-700 hover:to-purple-800
                      py-3.5 rounded-xl
                      shadow-lg hover:shadow-xl
                      transition-all duration-300
                      hover:-translate-y-1 active:scale-95
                    "
                  >
                    <ShoppingCart className="w-5 h-5" />
                    Add to Cart
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