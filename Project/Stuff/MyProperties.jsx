import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
// ✅ Import listingService to fetch your ads
import api, {
  propertyService,
  listingService,
  authService,
} from "../../services/api";
import Navbar from "../../components/Navbar";
import "./MyProperties.css";

// Icons
import ApartmentIcon from "@mui/icons-material/Apartment";
import ViewListIcon from "@mui/icons-material/ViewList";
import BedIcon from "@mui/icons-material/Bed";
import CircleIcon from "@mui/icons-material/Circle";

export default function MyProperties() {
  const navigate = useNavigate();
  const currentUser = authService.getCurrentUser();

  // --- STATES ---
  const [activeTab, setActiveTab] = useState("properties"); // 'properties' | 'listings'
  const [properties, setProperties] = useState([]);
  const [myListings, setMyListings] = useState([]); // Store fetched listings
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Modals
  const [openPropDialog, setOpenPropDialog] = useState(false);
  const [openRoomDialog, setOpenRoomDialog] = useState(false);
  const [selectedPropId, setSelectedPropId] = useState(null);

  // ✅ NEW: Property Details Modal State
  const [viewPropDetails, setViewPropDetails] = useState(null); // Stores the full property object (with rooms)

  // ✅ NEW: Edit Mode State
  const [isEditing, setIsEditing] = useState(false);
  const [editId, setEditId] = useState(null);

  // Forms
  // ✅ ADDED: image_url to state
  const [propForm, setPropForm] = useState({ address: "", description: "", image_url: "" });
  const [roomForm, setRoomForm] = useState({
    room_name: "",
    capacity: 1,
    area_m2: 0,
    base_rent: 0,
  });

  // --- FETCHING DATA ---
  const fetchData = async () => {
    setLoading(true);
    try {
      // 1. Fetch Properties
      const pRes = await propertyService.getAll({ owner_id: currentUser.id });
      setProperties(pRes.data.properties || []);

      // 2. Fetch Listings (For the new tab)
      const lRes = await listingService.getAll({ owner_id: currentUser.id });
      setMyListings(lRes.data.listings || []);
    } catch (err) {
      console.error(err);
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // --- HANDLERS ---

  // ✅ NEW: Handle Property Click (Fetch details + rooms)
  const handleViewProperty = async (propId) => {
    try {
      const res = await propertyService.getById(propId);
      // The controller returns { property, rooms, house_rules }
      setViewPropDetails(res.data);
    } catch (err) {
      alert("Failed to load property details");
    }
  };

  // ✅ UPDATED: Open Create Modal
  const openCreateModal = () => {
    setPropForm({ address: "", description: "", image_url: "" }); // Clear form
    setIsEditing(false);
    setOpenPropDialog(true);
  };

  // ✅ NEW: Open Edit Modal
  const openEditModal = (e, prop) => {
    e.stopPropagation(); // Stop card click
    setPropForm({ 
      address: prop.address, 
      description: prop.description || "", 
      image_url: prop.image_url || "" 
    });
    setEditId(prop.id);
    setIsEditing(true);
    setOpenPropDialog(true);
  };

  const handleDeleteProperty = async (e, id) => {
    e.stopPropagation(); // Stop card click
    if (
      !window.confirm(
        "Delete this property? This will also delete all rooms and listings attached to it."
      )
    )
      return;
    try {
      await propertyService.delete(id); // You need to ensure delete endpoint is exported in api.js
      fetchData(); // Refresh list
    } catch (err) {
      alert("Error deleting property");
    }
  };

  const handleDeleteListing = async (e, id) => {
    e.stopPropagation();
    if (
      !window.confirm(
        "Remove this listing? It will no longer be visible to tenants."
      )
    )
      return;
    try {
      await listingService.delete(id);
      fetchData(); // Refresh list
    } catch (err) {
      alert("Error deleting listing");
    }
  };

  // ✅ UPDATED: Handle Submit (Create or Update)
  const handlePropertyFormSubmit = async (e) => {
    e.preventDefault();
    try {
      if (isEditing) {
        // UPDATE Logic
        await propertyService.update(editId, propForm);
        alert("Property updated successfully!");
      } else {
        // CREATE Logic
        await propertyService.create(propForm);
        alert("Property created successfully!");
      }
      setOpenPropDialog(false);
      fetchData(); // Refresh list
    } catch (err) {
      console.error("Full Error Object:", err); // Check Console for details
      
      // ✅ FIX: Check for multiple error formats
      const errorMsg = 
        err.response?.data?.error ||                 // Standard error
        err.response?.data?.errors?.[0]?.msg ||      // Validation array
        err.message ||                               // Network error
        "Error saving property";
        
      alert(errorMsg);
    }
  };

  const handleCreateRoom = async (e) => {
    e.preventDefault();
    try {
      const payload = { ...roomForm, property_id: selectedPropId };
      await propertyService.createRoom(selectedPropId, payload);
      setOpenRoomDialog(false);
      setRoomForm({ room_name: "", capacity: 1, area_m2: 0, base_rent: 0 });
      alert("Room added successfully!");

      // Refresh the specific detail view if it's open
      if (viewPropDetails && viewPropDetails.property.id === selectedPropId) {
        handleViewProperty(selectedPropId);
      }
    } catch (err) {
      alert(
        "Error adding room: " + (err.response?.data?.error || "Unknown error")
      );
    }
  };

  if (loading && properties.length === 0)
    return (
      <>
        <Navbar title="My Properties" />
        <div className="loading-state">Loading your portfolio...</div>
      </>
    );

  return (
    <>
      <Navbar title="My Properties" />

      <div className="property-container">
        {/* HEADER & TABS */}
        <div className="page-header">
          <div>
            <h1 className="page-title">My Portfolio</h1>
            <p className="page-subtitle">
              Manage your buildings, rooms, and active ads.
            </p>
          </div>
          {/* ✅ UPDATED: Use openCreateModal */}
          <button
            className="btn btn-primary"
            onClick={openCreateModal}
          >
            + New Property
          </button>
        </div>

        {/* ✅ TAB SWITCHER */}
        <div className="tabs-container">
          <button
            className={`tab-btn ${activeTab === "properties" ? "active" : ""}`}
            onClick={() => setActiveTab("properties")}
          >
            <ApartmentIcon sx={{ fontSize: 18 }} /> Properties (
            {properties.length})
          </button>
          <button
            className={`tab-btn ${activeTab === "listings" ? "active" : ""}`}
            onClick={() => setActiveTab("listings")}
          >
            <ViewListIcon sx={{ fontSize: 18 }} /> Published Listings (
            {myListings.length})
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {/* --- VIEW 1: PROPERTIES GRID --- */}
        {activeTab === "properties" && (
          <div className="property-grid">
            {properties.length === 0 ? (
              <div className="empty-state">
                <p>No properties found.</p>
              </div>
            ) : (
              properties.map((prop) => (
                <div
                  className="property-card"
                  key={prop.id}
                  onClick={() => handleViewProperty(prop.id)}
                >
                  {/* ✅ NEW: Show Image on Card */}
                  <div style={{ height: '180px', overflow: 'hidden', borderRadius: '8px 8px 0 0', marginBottom: '1rem', marginLeft: '-1.5rem', marginRight: '-1.5rem', marginTop: '-1.5rem' }}>
                      <img 
                          src={prop.image_url || "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=500&q=60"} 
                          alt="Property" 
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                  </div>

                  {/* Clicking card now opens details */}
                  <div className="property-content clickable">
                    <div className="property-address">{prop.address}</div>
                    <div className="property-desc">
                      {prop.description || "No description provided."}
                    </div>
                    <div className="property-meta">
                      <span className="meta-tag">Click to view rooms</span>
                    </div>
                  </div>

                  <div
                    className="card-actions"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        setSelectedPropId(prop.id);
                        setOpenRoomDialog(true);
                      }}
                    >
                      + Room
                    </button>
                    
                    {/* ✅ NEW: Edit Button */}
                    <button 
                        className="btn btn-outline"
                        onClick={(e) => openEditModal(e, prop)}
                    >
                        ✏️
                    </button>

                    <button
                      className="btn btn-primary"
                      onClick={() =>
                        navigate(`/landlord/create-listing/${prop.id}`)
                      }
                    >
                      Post Ad
                    </button>
                    
                    {/* Delete Button */}
                    <button
                      className="btn btn-outline"
                      style={{ color: "#ef4444", borderColor: "#ef4444" }}
                      onClick={(e) => handleDeleteProperty(e, prop.id)}
                    >
                      🗑
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* --- VIEW 2: LISTINGS GRID (The new Tab) --- */}
        {activeTab === "listings" && (
          <div className="property-grid">
            {myListings.length === 0 ? (
              <div className="empty-state">
                <p>You haven't published any listings yet.</p>
                <button
                  className="btn btn-link"
                  onClick={() => setActiveTab("properties")}
                >
                  Go to Properties to create one
                </button>
              </div>
            ) : (
              myListings.map((listing) => (
                <div
                  className="property-card"
                  key={listing.id}
                  style={{
                    borderLeft:
                      listing.status === "verified"
                        ? "4px solid #10b981"
                        : "4px solid #f59e0b",
                  }}
                >
                  <div className="property-content">
                    <div className="listing-price">${listing.price} / mo</div>
                    <div
                      className="property-address"
                      style={{ fontSize: "0.95rem" }}
                    >
                      {listing.property_address}
                    </div>
                    <div className="listing-type">
                      {listing.room_name
                        ? `Room: ${listing.room_name}`
                        : "Whole Property"}
                    </div>
                    <div className={`status-badge ${listing.status}`}>
                      {listing.status.replace("_", " ")}
                    </div>
                  </div>
                  <div className="card-actions">
                    <button
                      className="btn btn-outline"
                      onClick={() => navigate(`/listings/${listing.id}`)}
                    >
                      Preview
                    </button>
                    <button
                      className="btn btn-outline"
                      style={{
                        color: "#ef4444",
                        borderColor: "#ef4444",
                        marginLeft: "10px",
                      }}
                      onClick={(e) => handleDeleteListing(e, listing.id)}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* --- MODAL 1: ADD / EDIT PROPERTY --- */}
        {openPropDialog && (
          <div
            className="modal-overlay"
            onClick={() => setOpenPropDialog(false)}
          >
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <h2 className="modal-title">
                {isEditing ? "Edit Property" : "Add New Property"}
              </h2>
              {/* ✅ UPDATED: Use handlePropertyFormSubmit */}
              <form onSubmit={handlePropertyFormSubmit}>
                <div className="form-group">
                  <label className="form-label">Full Address</label>
                  <input
                    type="text"
                    className="form-input"
                    required
                    autoFocus
                    value={propForm.address}
                    onChange={(e) =>
                      setPropForm({ ...propForm, address: e.target.value })
                    }
                  />
                </div>

                {/* ✅ NEW: Image URL Input */}
                <div className="form-group">
                  <label className="form-label">Property Image URL</label>
                  <input
                    type="url"
                    className="form-input"
                    placeholder="https://example.com/image.jpg"
                    value={propForm.image_url}
                    onChange={(e) => setPropForm({ ...propForm, image_url: e.target.value })}
                  />
                  {propForm.image_url && (
                    <img 
                      src={propForm.image_url} 
                      alt="Preview" 
                      style={{ width: '100%', height: '150px', objectFit: 'cover', marginTop: '10px', borderRadius: '8px' }} 
                      onError={(e) => e.target.style.display = 'none'} 
                    />
                  )}
                </div>

                <div className="form-group">
                  <label className="form-label">Description</label>
                  <textarea
                    className="form-input"
                    rows="3"
                    value={propForm.description}
                    onChange={(e) =>
                      setPropForm({ ...propForm, description: e.target.value })
                    }
                  />
                </div>
                <div className="modal-actions">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setOpenPropDialog(false)}
                  >
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary">
                    {isEditing ? "Save Changes" : "Create"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* --- MODAL 2: ADD ROOM --- */}
        {openRoomDialog && (
          <div
            className="modal-overlay"
            onClick={() => setOpenRoomDialog(false)}
          >
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <h2 className="modal-title">Add Room</h2>
              <form onSubmit={handleCreateRoom}>
                <div className="form-group">
                  <label className="form-label">Room Name</label>
                  <input
                    type="text"
                    className="form-input"
                    required
                    autoFocus
                    value={roomForm.room_name}
                    onChange={(e) =>
                      setRoomForm({ ...roomForm, room_name: e.target.value })
                    }
                  />
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "1rem",
                  }}
                >
                  <div className="form-group">
                    <label className="form-label">Capacity</label>
                    <input
                      type="number"
                      className="form-input"
                      required
                      min="1"
                      value={roomForm.capacity}
                      onChange={(e) =>
                        setRoomForm({ ...roomForm, capacity: e.target.value })
                      }
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Area (m²)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={roomForm.area_m2}
                      onChange={(e) =>
                        setRoomForm({ ...roomForm, area_m2: e.target.value })
                      }
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label className="form-label">Base Rent ($)</label>
                  <input
                    type="number"
                    className="form-input"
                    value={roomForm.base_rent}
                    onChange={(e) =>
                      setRoomForm({ ...roomForm, base_rent: e.target.value })
                    }
                  />
                </div>
                <div className="modal-actions">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setOpenRoomDialog(false)}
                  >
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary">
                    Save Room
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* --- MODAL 3: VIEW PROPERTY DETAILS (ROOMS) --- */}
        {viewPropDetails && (
          <div
            className="modal-overlay"
            onClick={() => setViewPropDetails(null)}
          >
            <div
              className="modal-content large"
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <h2 className="modal-title">
                  {viewPropDetails.property.address}
                </h2>
                <button
                  className="btn-close"
                  onClick={() => setViewPropDetails(null)}
                >
                  ×
                </button>
              </div>

              <p style={{ color: "#666", marginBottom: "1.5rem" }}>
                {viewPropDetails.property.description || "No description."}
              </p>

              <h3
                style={{
                  fontSize: "1rem",
                  fontWeight: "600",
                  marginBottom: "1rem",
                }}
              >
                Rooms in this property
              </h3>

              {!viewPropDetails.rooms || viewPropDetails.rooms.length === 0 ? (
                <div className="empty-state small">No rooms added yet.</div>
              ) : (
                <div className="rooms-list">
                  {viewPropDetails.rooms.map((room) => (
                    <div key={room.id} className="room-item">
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "10px",
                        }}
                      >
                        <BedIcon sx={{ color: "#94a3b8" }} />
                        <div>
                          <div style={{ fontWeight: "500" }}>
                            {room.room_name}
                          </div>
                          <div style={{ fontSize: "0.8rem", color: "#666" }}>
                            {room.capacity} People • {room.area_m2}m²
                          </div>
                        </div>
                      </div>
                      <div style={{ fontWeight: "600", color: "#333" }}>
                        ${room.base_rent || 0}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="modal-actions" style={{ marginTop: "2rem" }}>
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    setSelectedPropId(viewPropDetails.property.id);
                    setViewPropDetails(null); // Close detail view
                    setOpenRoomDialog(true); // Open add room dialog
                  }}
                >
                  + Add Another Room
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}