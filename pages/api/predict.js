import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { symptoms } = req.body;

    if (!symptoms || !Array.isArray(symptoms)) {
      return res.status(400).json({ message: 'Invalid symptoms input' });
    }

    // Call the Python model service
    const response = await axios.post('http://localhost:5000/predict', { symptoms });
    
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      message: 'Error processing prediction',
      error: error.message 
    });
  }
} 