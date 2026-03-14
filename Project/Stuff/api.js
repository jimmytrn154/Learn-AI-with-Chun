import axios from 'axios';

// ⚠️ IMPORTANT: If you moved your backend to port 5000, change this URL to 'http://localhost:5000/api'
const API_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor: Attaches token to every request automatically
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const authService = {
  login: async (email, password) => {
    const response = await api.post('/auth/login', { email, password });
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    return response.data;
  },

  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login';
  },

  getCurrentUser: () => {
    const userStr = localStorage.getItem('user');
    if (userStr) return JSON.parse(userStr);
    return null;
  },
};

// --- PROPERTY SERVICE ---
export const propertyService = {
  getAll: (params) => api.get('/properties', { params }), // Updated to accept params
  getById: (id) => api.get(`/properties/${id}`),
  create: (data) => api.post('/properties', data),
  // Note: Your properties.js route for rooms is router.post('/:id/rooms', ...)
  update: (id, data) => api.put(`/properties/${id}`, data), 
  delete: (id) => api.delete(`/properties/${id}`),
  createRoom: (propertyId, data) => api.post(`/properties/${propertyId}/rooms`, data),
};

// --- LISTING SERVICE ---
export const listingService = {
  // Supports filters like: getAll({ status: 'verified', min_price: 1000 })
  getAll: (params) => api.get('/listings', { params }),
  getById: (id) => api.get(`/listings/${id}`),
  create: (data) => api.post('/listings', data),
  update: (id, data) => api.put(`/listings/${id}`, data),
  delete: (id) => api.delete(`/listings/${id}`),
};

// --- RENTAL REQUEST SERVICE ---
export const requestService = {
  getAll: () => api.get('/rental-requests'),
  getById: (id) => api.get(`/rental-requests/${id}`),
  create: (data) => api.post('/rental-requests', data),
  updateStatus: (id, status) => api.put(`/rental-requests/${id}/status`, { status }),
};

// --- CONTRACT SERVICE ---
export const contractService = {
  getAll: () => api.get('/contracts'),
  getById: (id) => api.get(`/contracts/${id}`),
  sign: (id) => api.post(`/contracts/${id}/sign`),
};

// --- ISSUE SERVICE ---
export const issueService = {
  getAll: () => api.get('/issues'),
  getById: (id) => api.get(`/issues/${id}`),
  create: (data) => api.post('/issues', data),
  updateStatus: (id, status) => api.put(`/issues/${id}/status`, { status }),
};

// --- VERIFICATION SERVICE ---
export const verificationService = {
  getAll: (params) => api.get('/verifications', { params }),
  getById: (id) => api.get(`/verifications/${id}`),
  create: (data) => api.post('/verifications', data),
  update: (id, data) => api.put(`/verifications/${id}`, data),
};

// --- USER SERVICE ---
export const userService = {
  getAll: (params) => api.get('/users', { params }),
  getById: (id) => api.get(`/users/${id}`),
  updateStatus: (id, status) => api.put(`/users/${id}/status`, { status }),
};

// --- ORGANIZATION SERVICE ---
export const organizationService = {
  getAll: () => api.get('/organizations'),
  create: (data) => api.post('/organizations', data),
  updateAffiliation: (userId, orgId, status) => api.put(`/organizations/affiliations/${userId}/${orgId}`, { status }),
};

export default api;