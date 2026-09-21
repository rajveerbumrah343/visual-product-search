import React, { useState, useCallback } from "react";

const API_BASE = ""; // same-origin via Vite proxy; set to your API URL in production

export default function App() {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryTimeMs, setQueryTimeMs] = useState(null);

  const handleFileChange = useCallback((file) => {
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults([]);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      handleFileChange(file);
    },
    [handleFileChange]
  );

  const handleSearch = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE}/api/search?top_k=8`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed with status ${res.status}`);
      }

      const data = await res.json();
      setResults(data.results);
      setQueryTimeMs(data.query_time_ms);
    } catch (err) {
      setError(err.message || "Something went wrong while searching.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>Visual Product Search</h1>
        <p style={styles.subtitle}>
          Upload a product photo to find visually similar items in the catalog.
        </p>
      </header>

      <div
        style={styles.dropzone}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="preview" style={styles.previewImg} />
        ) : (
          <p style={styles.dropText}>Drag & drop an image here, or click below to choose one</p>
        )}
      </div>

      <div style={styles.controls}>
        <label style={styles.fileButton}>
          Choose Image
          <input
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFileChange(e.target.files?.[0])}
          />
        </label>

        <button
          onClick={handleSearch}
          disabled={!selectedFile || loading}
          style={{
            ...styles.searchButton,
            opacity: !selectedFile || loading ? 0.5 : 1,
          }}
        >
          {loading ? "Searching..." : "Find Similar Products"}
        </button>
      </div>

      {error && <p style={styles.error}>{error}</p>}

      {queryTimeMs !== null && !error && (
        <p style={styles.meta}>
          Found {results.length} result{results.length !== 1 ? "s" : ""} in {queryTimeMs} ms
        </p>
      )}

      <div style={styles.grid}>
        {results.map((product) => (
          <ProductCard key={product.product_id} product={product} />
        ))}
      </div>
    </div>
  );
}

function ProductCard({ product }) {
  const similarityPct = (product.similarity * 100).toFixed(1);
  return (
    <div style={styles.card}>
      <img src={product.image_path} alt={product.name} style={styles.cardImg} />
      <div style={styles.cardBody}>
        <h3 style={styles.cardTitle}>{product.name}</h3>
        {product.category && <p style={styles.cardCategory}>{product.category}</p>}
        {product.price != null && <p style={styles.cardPrice}>${product.price.toFixed(2)}</p>}
        <div style={styles.similarityBarTrack}>
          <div
            style={{
              ...styles.similarityBarFill,
              width: `${Math.max(0, Math.min(100, similarityPct))}%`,
            }}
          />
        </div>
        <p style={styles.similarityLabel}>{similarityPct}% visually similar</p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    maxWidth: 960,
    margin: "0 auto",
    padding: "32px 16px",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    color: "#1a1a1a",
  },
  header: { textAlign: "center", marginBottom: 24 },
  title: { fontSize: 28, fontWeight: 700, margin: 0 },
  subtitle: { color: "#666", marginTop: 8 },
  dropzone: {
    border: "2px dashed #ccc",
    borderRadius: 12,
    minHeight: 220,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#fafafa",
    overflow: "hidden",
  },
  dropText: { color: "#888" },
  previewImg: { maxHeight: 220, maxWidth: "100%", objectFit: "contain" },
  controls: {
    display: "flex",
    gap: 12,
    justifyContent: "center",
    marginTop: 16,
  },
  fileButton: {
    padding: "10px 18px",
    borderRadius: 8,
    background: "#eee",
    cursor: "pointer",
    fontWeight: 600,
  },
  searchButton: {
    padding: "10px 18px",
    borderRadius: 8,
    background: "#1a73e8",
    color: "white",
    border: "none",
    fontWeight: 600,
    cursor: "pointer",
  },
  error: { color: "#c0392b", textAlign: "center", marginTop: 16 },
  meta: { color: "#666", textAlign: "center", marginTop: 16, fontSize: 14 },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
    gap: 16,
    marginTop: 24,
  },
  card: {
    border: "1px solid #eee",
    borderRadius: 12,
    overflow: "hidden",
    background: "white",
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
  },
  cardImg: { width: "100%", height: 160, objectFit: "cover" },
  cardBody: { padding: 12 },
  cardTitle: { fontSize: 14, fontWeight: 600, margin: "0 0 4px" },
  cardCategory: { fontSize: 12, color: "#888", margin: "0 0 4px" },
  cardPrice: { fontSize: 13, fontWeight: 600, margin: "0 0 8px" },
  similarityBarTrack: {
    height: 6,
    borderRadius: 3,
    background: "#eee",
    overflow: "hidden",
  },
  similarityBarFill: {
    height: "100%",
    background: "#1a73e8",
  },
  similarityLabel: { fontSize: 11, color: "#666", marginTop: 4 },
};
