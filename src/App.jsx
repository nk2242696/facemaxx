import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, ArrowRight, ArrowLeft, Sparkles, Loader, AlertCircle } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

export default function FacialAnalysisApp() {
    const [step, setStep] = useState(1);
    const [image, setImage] = useState(null);
    const [scores, setScores] = useState(null);
    const [loading, setLoading] = useState(false);
    const [model, setModel] = useState(null);
    const [modelLoading, setModelLoading] = useState(false);
    const [modelError, setModelError] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [cameraStatus, setCameraStatus] = useState('idle'); // 'idle', 'starting', 'ready', 'error'
    const [aiRecommendations, setAiRecommendations] = useState({});
    const [loadingRecommendations, setLoadingRecommendations] = useState(false);
    const [expandedRecommendation, setExpandedRecommendation] = useState(null);
    const [detailedGuide, setDetailedGuide] = useState(null);
    const [loadingDetailedGuide, setLoadingDetailedGuide] = useState(false);
    const [theme, setTheme] = useState('teal'); // 'teal', 'purple', 'blue', 'green', 'rose'
    const [showThemeSelector, setShowThemeSelector] = useState(false);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const imageRef = useRef(null);
    const [stream, setStream] = useState(null);

    // Theme configurations
    const themes = {
        teal: {
            name: 'Ocean Breeze',
            bg: 'from-slate-950 via-teal-950 to-slate-950',
            primary: 'from-teal-500 to-cyan-600',
            secondary: 'from-orange-500 to-orange-600',
            accent: 'teal-400',
            gradient: 'from-teal-500 via-cyan-600 to-orange-500',
            orb1: 'bg-teal-500',
            orb2: 'bg-cyan-500',
            orb3: 'bg-orange-500',
            glow: 'shadow-teal-500/30',
            text: 'from-white via-teal-200 to-cyan-200',
            border: 'border-teal-500/30',
            hover: 'hover:shadow-teal-500/20 hover:border-teal-500/30',
            excellent: 'from-teal-400 to-cyan-500',
        },
        purple: {
            name: 'Royal Purple',
            bg: 'from-slate-950 via-purple-950 to-slate-950',
            primary: 'from-purple-500 to-pink-600',
            secondary: 'from-pink-500 to-rose-600',
            accent: 'purple-400',
            gradient: 'from-purple-500 via-pink-600 to-fuchsia-500',
            orb1: 'bg-purple-500',
            orb2: 'bg-pink-500',
            orb3: 'bg-fuchsia-500',
            glow: 'shadow-purple-500/30',
            text: 'from-white via-purple-200 to-pink-200',
            border: 'border-purple-500/30',
            hover: 'hover:shadow-purple-500/20 hover:border-purple-500/30',
            excellent: 'from-purple-400 to-pink-500',
        },
        blue: {
            name: 'Deep Ocean',
            bg: 'from-slate-950 via-blue-950 to-slate-950',
            primary: 'from-blue-500 to-indigo-600',
            secondary: 'from-sky-500 to-blue-600',
            accent: 'blue-400',
            gradient: 'from-blue-500 via-indigo-600 to-sky-500',
            orb1: 'bg-blue-500',
            orb2: 'bg-indigo-500',
            orb3: 'bg-sky-500',
            glow: 'shadow-blue-500/30',
            text: 'from-white via-blue-200 to-indigo-200',
            border: 'border-blue-500/30',
            hover: 'hover:shadow-blue-500/20 hover:border-blue-500/30',
            excellent: 'from-blue-400 to-indigo-500',
        },
        green: {
            name: 'Forest Mint',
            bg: 'from-slate-950 via-emerald-950 to-slate-950',
            primary: 'from-emerald-500 to-green-600',
            secondary: 'from-lime-500 to-green-600',
            accent: 'emerald-400',
            gradient: 'from-emerald-500 via-green-600 to-lime-500',
            orb1: 'bg-emerald-500',
            orb2: 'bg-green-500',
            orb3: 'bg-lime-500',
            glow: 'shadow-emerald-500/30',
            text: 'from-white via-emerald-200 to-green-200',
            border: 'border-emerald-500/30',
            hover: 'hover:shadow-emerald-500/20 hover:border-emerald-500/30',
            excellent: 'from-emerald-400 to-green-500',
        },
        rose: {
            name: 'Sunset Rose',
            bg: 'from-slate-950 via-rose-950 to-slate-950',
            primary: 'from-rose-500 to-amber-600',
            secondary: 'from-amber-500 to-orange-600',
            accent: 'rose-400',
            gradient: 'from-rose-500 via-amber-600 to-orange-500',
            orb1: 'bg-rose-500',
            orb2: 'bg-amber-500',
            orb3: 'bg-orange-500',
            glow: 'shadow-rose-500/30',
            text: 'from-white via-rose-200 to-amber-200',
            border: 'border-rose-500/30',
            hover: 'hover:shadow-rose-500/20 hover:border-rose-500/30',
            excellent: 'from-rose-400 to-amber-500',
        },
    };

    const currentTheme = themes[theme];

    // Monitor camera status changes for debugging
    useEffect(() => {
        console.log('🔄 Camera status changed to:', cameraStatus);
    }, [cameraStatus]);

    // Fetch detailed guide when modal opens
    useEffect(() => {
        if (expandedRecommendation && !detailedGuide) {
            const fetchGuide = async () => {
                setLoadingDetailedGuide(true);
                try {
                    const guide = await getDetailedAIGuide(
                        expandedRecommendation.feature,
                        expandedRecommendation.recommendation,
                        expandedRecommendation.score
                    );
                    setDetailedGuide(guide);
                } catch (error) {
                    console.error('Error fetching detailed guide:', error);
                } finally {
                    setLoadingDetailedGuide(false);
                }
            };
            fetchGuide();
        } else if (!expandedRecommendation) {
            setDetailedGuide(null); // Reset when modal closes
        }
    }, [expandedRecommendation]);

    // Load model in background when component mounts, but don't block UI
    useEffect(() => {
        const preloadModel = async () => {
            try {
                // Add timeout to prevent hanging
                const timeoutPromise = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('Model loading timeout')), 30000); // 30 second timeout
                });
                
                await Promise.race([loadModel(), timeoutPromise]);
            } catch (error) {
                console.log('Pre-loading model failed, will try again when needed:', error.message);
            }
        };
        preloadModel();
    }, []);

    const loadModel = async () => {
        if (model || modelLoading) return model; // Return existing model or prevent multiple loads
        
        setModelLoading(true);
        setModelError(null);
        
        try {
            console.log('Starting AI model initialization...');
            console.log('Available faceLandmarksDetection:', Object.keys(faceLandmarksDetection));
            console.log('faceLandmarksDetection object:', faceLandmarksDetection);
            
            // Set TensorFlow.js backend preference for better performance
            await tf.ready();
            
            // Try to use WebGL backend for better performance
            if (tf.getBackend() !== 'webgl') {
                try {
                    await tf.setBackend('webgl');
                } catch (e) {
                    console.log('WebGL backend not available, using default backend');
                }
            }
            
            console.log('TensorFlow.js ready with backend:', tf.getBackend(), 'loading face detection model...');
            
            let loadedModel;
            
            // Use the stable v0.0.3 API
            console.log('Loading face landmarks detection model with v0.0.3 API...');
            loadedModel = await faceLandmarksDetection.load(
                faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
                { maxFaces: 1 }
            );


            
            console.log('Model loaded successfully with backend:', tf.getBackend());
            setModel(loadedModel);
            setModelLoading(false);
            return loadedModel;
        } catch (error) {
            console.error('All model loading strategies failed:', error);
            
            const isNetworkError = error.message.includes('fetch') || 
                                 error.message.includes('network') || 
                                 error.message.includes('Failed to fetch') ||
                                 error.message.includes('NetworkError');
            
            const errorMessage = isNetworkError 
                ? 'Network connection error. Please check your internet and try again.'
                : `Model loading failed: ${error.message}. Try refreshing the page.`;
                
            setModelError(errorMessage);
            setModelLoading(false);
            throw error;
        }
    };

    const calculateDistance = (point1, point2) => {
        return Math.sqrt(
            Math.pow(point1[0] - point2[0], 2) +
            Math.pow(point1[1] - point2[1], 2)
        );
    };

    const calculateAngle = (p1, p2, p3) => {
        const a = calculateDistance(p2, p3);
        const b = calculateDistance(p1, p3);
        const c = calculateDistance(p1, p2);
        return Math.acos((a * a + c * c - b * b) / (2 * a * c)) * (180 / Math.PI);
    };

    // AI-based facial feature calculation functions
    const calculateFaceShape = (faceRatio, jawRatio, cheekboneRatio, landmarks) => {
        // Add some randomization based on specific landmark positions to ensure uniqueness
        const uniqueFactor = (landmarks[10][0] + landmarks[152][1] + landmarks[234][0]) % 100 / 1000;
        const adjustedFaceRatio = faceRatio + uniqueFactor;
        const adjustedJawRatio = jawRatio + (uniqueFactor * 0.1);
        const adjustedCheekboneRatio = cheekboneRatio + (uniqueFactor * 0.08);
        
        console.log('Face Shape Analysis:', {
            original: { faceRatio, jawRatio, cheekboneRatio },
            adjusted: { adjustedFaceRatio, adjustedJawRatio, adjustedCheekboneRatio },
            uniqueFactor
        });
        
        const jawCheekboneDiff = Math.abs(adjustedJawRatio - adjustedCheekboneRatio);
        
        // More nuanced classification with tighter bounds
        if (adjustedFaceRatio > 1.38) {
            return 'Oblong';
        } else if (adjustedFaceRatio < 1.12) {
            return jawCheekboneDiff < 0.03 ? 'Square' : 'Round';
        } else if (adjustedFaceRatio > 1.28) {
            return adjustedCheekboneRatio > adjustedJawRatio + 0.02 ? 'Heart' : 'Oval';
        } else if (adjustedFaceRatio > 1.18) {
            return adjustedJawRatio > adjustedCheekboneRatio + 0.03 ? 'Diamond' : 'Rectangular';
        } else {
            return jawCheekboneDiff > 0.04 ? 'Triangle' : 'Oval';
        }
    };

    const calculateEyeType = (eyeToFaceRatio, leftEyeWidth, rightEyeWidth, eyeDistance, landmarks) => {
        const avgEyeWidth = (leftEyeWidth + rightEyeWidth) / 2;
        const eyeSpacing = eyeDistance / avgEyeWidth;
        
        // Add unique factors based on actual eye landmark positions
        const eyeAsymmetry = Math.abs(leftEyeWidth - rightEyeWidth) / avgEyeWidth;
        const eyeHeightL = Math.abs(landmarks[159][1] - landmarks[145][1]); // Left eye height
        const eyeHeightR = Math.abs(landmarks[386][1] - landmarks[374][1]); // Right eye height
        const avgEyeHeight = (eyeHeightL + eyeHeightR) / 2;
        const eyeAspectRatio = avgEyeWidth / avgEyeHeight;
        
        console.log('Eye Analysis:', {
            eyeToFaceRatio, eyeSpacing, eyeAsymmetry, eyeAspectRatio,
            leftEyeWidth, rightEyeWidth, eyeDistance
        });
        
        // More dynamic classification
        if (eyeToFaceRatio < 0.27 && eyeAspectRatio > 3.5) {
            return 'Hunter Eyes';
        } else if (eyeToFaceRatio < 0.30 && eyeSpacing > 3.3) {
            return 'Narrow';
        } else if (eyeToFaceRatio > 0.36) {
            return eyeSpacing < 2.7 ? 'Large & Close' : 'Large';
        } else if (eyeAspectRatio > 4.0) {
            return 'Elongated';
        } else if (eyeAspectRatio < 3.0) {
            return 'Round';
        } else {
            return eyeAsymmetry > 0.1 ? 'Asymmetric' : 'Almond';
        }
    };

    const calculateCanthalTilt = (landmarks) => {
        // More precise canthal tilt calculation
        // Left eye: outer corner (33) to inner corner (133)
        // Right eye: inner corner (362) to outer corner (263)
        const leftOuter = landmarks[33];
        const leftInner = landmarks[133];
        const rightInner = landmarks[362];
        const rightOuter = landmarks[263];
        
        // Calculate actual angles instead of just Y difference
        const leftTiltAngle = Math.atan2(leftOuter[1] - leftInner[1], leftOuter[0] - leftInner[0]) * (180 / Math.PI);
        const rightTiltAngle = Math.atan2(rightOuter[1] - rightInner[1], rightOuter[0] - rightInner[0]) * (180 / Math.PI);
        const avgTiltAngle = (leftTiltAngle + rightTiltAngle) / 2;
        
        // Add some variability based on landmark positions
        const variabilityFactor = (leftOuter[0] + rightOuter[0] + leftInner[1] + rightInner[1]) % 10 - 5;
        const adjustedTilt = avgTiltAngle + (variabilityFactor * 0.5);
        
        console.log('Canthal Tilt Analysis:', {
            leftTiltAngle, rightTiltAngle, avgTiltAngle, adjustedTilt, variabilityFactor
        });
        
        if (adjustedTilt < -3) {
            return 'Positive';
        } else if (adjustedTilt > 3) {
            return 'Negative';
        } else if (Math.abs(adjustedTilt) < 1) {
            return 'Neutral';
        } else {
            return adjustedTilt < 0 ? 'Slightly Positive' : 'Slightly Negative';
        }
    };

    const calculateMaxillaDevelopment = (cheekboneRatio, landmarks) => {
        // More comprehensive maxilla analysis
        const cheekboneL = landmarks[205];  // Left cheekbone
        const cheekboneR = landmarks[425];  // Right cheekbone
        const noseBase = landmarks[2];      // Nose base
        const lipTop = landmarks[13];       // Upper lip
        const chin = landmarks[152];        // Chin
        
        // Calculate multiple projection metrics
        const avgCheekboneY = (cheekboneL[1] + cheekboneR[1]) / 2;
        const midFaceHeight = Math.abs(noseBase[1] - avgCheekboneY);
        const lowerFaceHeight = Math.abs(chin[1] - lipTop[1]);
        const midToLowerRatio = midFaceHeight / lowerFaceHeight;
        
        // Add variability based on actual measurements
        const cheekboneWidth = Math.abs(cheekboneL[0] - cheekboneR[0]);
        const jawWidth = Math.abs(landmarks[234][0] - landmarks[454][0]);
        const cheekboneToJawRatio = cheekboneWidth / jawWidth;
        
        console.log('Maxilla Analysis:', {
            cheekboneRatio, midFaceHeight, lowerFaceHeight, midToLowerRatio,
            cheekboneToJawRatio, cheekboneWidth, jawWidth
        });
        
        // More nuanced classification
        if (cheekboneRatio > 0.70 && midToLowerRatio > 1.1 && cheekboneToJawRatio > 1.05) {
            return 'Very Strong';
        } else if (cheekboneRatio > 0.65 && midToLowerRatio > 0.9) {
            return 'Strong';
        } else if (cheekboneRatio < 0.58 || midToLowerRatio < 0.7) {
            return 'Weak';
        } else if (cheekboneRatio < 0.62 && midToLowerRatio < 0.85) {
            return 'Below Average';
        } else {
            return 'Neutral';
        }
    };

    const calculateNoseShape = (landmarks, noseRatio, faceRatio) => {
        // More detailed nose analysis using multiple landmarks
        const noseTip = landmarks[1];       // Nose tip
        const noseBridge = landmarks[6];    // Nose bridge
        const noseTop = landmarks[19];      // Nose top
        const noseLeft = landmarks[98];     // Left nostril
        const noseRight = landmarks[327];   // Right nostril
        const noseBottom = landmarks[2];    // Nose bottom/base
        
        // Calculate comprehensive nose metrics
        const noseHeight = Math.abs(noseTop[1] - noseBottom[1]);
        const noseWidth = Math.abs(noseLeft[0] - noseRight[0]);
        const bridgeCurvature = Math.abs(noseBridge[1] - ((noseTop[1] + noseTip[1]) / 2));
        const tipProjection = Math.abs(noseTip[1] - noseBottom[1]);
        const nostrilAsymmetry = Math.abs(noseLeft[1] - noseRight[1]);
        
        // Add uniqueness based on landmark positions
        const uniqueFactor = (noseTip[0] + noseBridge[1] + noseLeft[0]) % 20 / 100;
        const adjustedRatio = noseRatio + uniqueFactor;
        
        console.log('Nose Analysis:', {
            noseRatio: adjustedRatio, noseHeight, noseWidth, bridgeCurvature,
            tipProjection, nostrilAsymmetry, uniqueFactor
        });
        
        // More sophisticated classification
        if (adjustedRatio > 0.26) {
            return noseHeight > 35 ? 'Wide & Long' : bridgeCurvature > 5 ? 'Wide & Curved' : 'Wide';
        } else if (adjustedRatio < 0.17) {
            return tipProjection < 8 ? 'Button' : bridgeCurvature > 4 ? 'Narrow & Aquiline' : 'Narrow';
        } else if (bridgeCurvature > 6) {
            return faceRatio > 1.3 ? 'Roman' : 'Curved';
        } else if (noseHeight > 40) {
            return nostrilAsymmetry > 2 ? 'Long & Asymmetric' : 'Long';
        } else if (tipProjection < 10) {
            return 'Upturned';
        } else {
            return nostrilAsymmetry > 1.5 ? 'Slightly Asymmetric' : 'Straight';
        }
    };

    const basicImageAnalysis = (imageSrc) => {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                // Basic image analysis without AI model
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // Simple image quality metrics
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const pixels = imageData.data;
                let brightness = 0;
                let variance = 0;
                
                for (let i = 0; i < pixels.length; i += 4) {
                    const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
                    brightness += gray;
                }
                brightness = brightness / (pixels.length / 4);
                
                // Calculate deterministic metrics based on image properties
                let avgR = 0, avgG = 0, avgB = 0;
                let rVariance = 0, gVariance = 0, bVariance = 0;
                
                // First pass: calculate averages
                for (let i = 0; i < pixels.length; i += 4) {
                    avgR += pixels[i];
                    avgG += pixels[i + 1];
                    avgB += pixels[i + 2];
                }
                avgR /= (pixels.length / 4);
                avgG /= (pixels.length / 4);
                avgB /= (pixels.length / 4);
                
                // Second pass: calculate variance for texture analysis
                for (let i = 0; i < pixels.length; i += 4) {
                    rVariance += Math.pow(pixels[i] - avgR, 2);
                    gVariance += Math.pow(pixels[i + 1] - avgG, 2);
                    bVariance += Math.pow(pixels[i + 2] - avgB, 2);
                }
                rVariance = Math.sqrt(rVariance / (pixels.length / 4));
                gVariance = Math.sqrt(gVariance / (pixels.length / 4));
                bVariance = Math.sqrt(bVariance / (pixels.length / 4));
                
                // Use deterministic image properties for scoring
                const baseScore = Math.max(45, Math.min(85, (brightness + avgR + avgG + avgB) / 12));
                const contrast = Math.abs(avgR - avgG) + Math.abs(avgG - avgB) + Math.abs(avgB - avgR);
                const texture = (rVariance + gVariance + bVariance) / 3;
                const colorBalance = Math.min(avgR, avgG, avgB) / Math.max(avgR, avgG, avgB);
                
                // Create deterministic hash from image properties for consistent variation
                const imageSignature = Math.floor((avgR * 7 + avgG * 13 + avgB * 17 + contrast * 3 + texture * 2) * 1000) % 1000;
                
                // Create varied scores using deterministic image properties
                const createVariedScore = (base, seedMultiplier, min = 35, max = 95) => {
                    const seed = (imageSignature * seedMultiplier) % 100;
                    const variation = (seed / 100) * 30 - 15; // -15 to +15 variation
                    const textureBonus = Math.min(10, texture / 10); // Texture adds up to 10 points
                    const contrastBonus = Math.min(8, contrast / 20); // Contrast adds up to 8 points
                    
                    return Math.round(Math.max(min, Math.min(max, base + variation + textureBonus + contrastBonus)));
                };
                
                // Calculate final scores using deterministic methods
                const skinScore = Math.round(Math.max(30, Math.min(95, 
                    100 - (255 - brightness) * 0.4 + contrast * 0.1 + colorBalance * 15
                )));
                
                const overallScore = Math.round(
                    (createVariedScore(baseScore, 7) + skinScore) / 2
                );
                
                resolve({
                    current: overallScore,
                    potential: Math.round(Math.min(overallScore + 12, 92)),
                    isBasicMode: true,
                    features: {
                        symmetry: createVariedScore(baseScore, 7),
                        jawline: createVariedScore(baseScore, 11),
                        cheekbones: createVariedScore(baseScore, 9),
                        noseShape: createVariedScore(baseScore, 13),
                        eyeArea: createVariedScore(baseScore, 8),
                        faceShape: createVariedScore(baseScore, 12),
                        forehead: createVariedScore(baseScore, 6),
                        masculinity: createVariedScore(baseScore, 14),
                        skinQuality: skinScore,
                        hairstyle: createVariedScore(baseScore, 15, 40, 90)
                    },
                    measurements: {
                        note: "⚠️ Basic estimation mode - AI model not loaded. Results are deterministic based on image properties.",
                        brightness: Math.round(brightness),
                        contrast: Math.round(contrast),
                        texture: Math.round(texture),
                        colorBalance: Math.round(colorBalance * 100) + '%'
                    }
                });
            };
            img.src = imageSrc;
        });
    };

    const analyzeFacialFeatures = async (imageSrc) => {
        setLoading(true);
        
        try {
            // Load model if not already loaded
            let currentModel = model;
            if (!currentModel && !modelError) {
                console.log('Model not loaded, loading now...');
                try {
                    currentModel = await loadModel();
                } catch (error) {
                    console.log('Failed to load model, using basic analysis');
                }
            }

            // If no model available, use basic analysis
            if (!currentModel) {
                console.log('AI model not available, using basic image analysis mode');
                const basicResults = await basicImageAnalysis(imageSrc);
                setLoading(false);
                return basicResults;
            }

            const img = new Image();
            img.src = imageSrc;

            await new Promise((resolve) => {
                img.onload = resolve;
            });

            const predictions = await currentModel.estimateFaces({
                input: img,
                returnTensors: false,
                flipHorizontal: false,
                predictIrises: true
            });

            if (predictions.length === 0) {
                alert('No face detected. Please try another photo.');
                setLoading(false);
                return null;
            }

            const landmarks = predictions[0].scaledMesh;

            // Key landmark indices based on MediaPipe Face Mesh
            const indices = {
                leftEye: [33, 133, 160, 144],
                rightEye: [362, 263, 387, 373],
                nose: [1, 2, 98, 327],
                mouth: [61, 291, 0, 17],
                jawline: [234, 454, 152, 378],
                forehead: [10, 338, 297, 109],
                cheekbones: [205, 425, 50, 280],
                chin: [152]
            };

            // Calculate facial measurements
            const faceWidth = calculateDistance(landmarks[234], landmarks[454]);
            const faceHeight = calculateDistance(landmarks[10], landmarks[152]);
            const noseWidth = calculateDistance(landmarks[98], landmarks[327]);
            const mouthWidth = calculateDistance(landmarks[61], landmarks[291]);
            const jawWidth = calculateDistance(landmarks[234], landmarks[454]);
            const foreheadWidth = calculateDistance(landmarks[109], landmarks[338]);

            // Eye measurements
            const leftEyeWidth = calculateDistance(landmarks[33], landmarks[133]);
            const rightEyeWidth = calculateDistance(landmarks[362], landmarks[263]);
            const eyeDistance = calculateDistance(landmarks[133], landmarks[362]);

            // Canthal tilt calculation (angle between outer and inner eye corners)
            const leftEyeAngle = Math.atan2(landmarks[133][1] - landmarks[33][1], landmarks[133][0] - landmarks[33][0]);
            const rightEyeAngle = Math.atan2(landmarks[362][1] - landmarks[263][1], landmarks[362][0] - landmarks[263][0]);

            // Cheekbone measurements
            const cheekboneWidth = calculateDistance(landmarks[205], landmarks[425]);

            // Calculate symmetry
            const leftFacePoints = [33, 130, 234, 127];
            const rightFacePoints = [362, 359, 454, 356];
            let symmetryScore = 0;

            for (let i = 0; i < leftFacePoints.length; i++) {
                const leftDist = Math.abs(landmarks[leftFacePoints[i]][0] - landmarks[1][0]);
                const rightDist = Math.abs(landmarks[rightFacePoints[i]][0] - landmarks[1][0]);
                const diff = Math.abs(leftDist - rightDist);
                symmetryScore += (1 - Math.min(diff / faceWidth, 1)) * 25;
            }

            // WEIGHTED SCORING SYSTEM

            // 1. Symmetry Score (0-100)
            const finalSymmetry = Math.min(symmetryScore, 100);

            // 2. Jawline Score - wider jaw at bottom, defined angle
            const jawToFaceRatio = jawWidth / faceWidth;
            const jawlineScore = Math.min(
                (jawToFaceRatio > 0.85 ? (jawToFaceRatio - 0.85) * 400 : 0) + 60,
                100
            );

            // 3. Cheekbone Score - high and prominent
            const cheekboneToFaceRatio = cheekboneWidth / faceWidth;
            const cheekboneScore = Math.min(
                (cheekboneToFaceRatio > 0.95 ? (cheekboneToFaceRatio - 0.95) * 800 : 0) + 65,
                100
            );

            // 4. Nose Score - thinner nose is more desirable
            const noseToFaceRatio = noseWidth / faceWidth;
            const noseScore = Math.max(
                100 - (noseToFaceRatio - 0.15) * 400,
                40
            );

            // 5. Eye Area Score - larger eyes, good spacing
            const avgEyeWidth = (leftEyeWidth + rightEyeWidth) / 2;
            const eyeToFaceRatio = avgEyeWidth / faceWidth;
            const eyeSpacingRatio = eyeDistance / faceWidth;
            const eyeScore = Math.min(
                (eyeToFaceRatio * 400) + (eyeSpacingRatio > 0.4 ? 20 : 0) + 40,
                100
            );

            // 6. Face Shape Score - golden ratio considerations
            const faceRatio = faceHeight / faceWidth;
            const goldenRatio = 1.618;
            const ratioScore = Math.max(
                100 - Math.abs(faceRatio - goldenRatio) * 60,
                50
            );

            // 7. Forehead Score - wider forehead is positive
            const foreheadToFaceRatio = foreheadWidth / faceWidth;
            const foreheadScore = Math.min(
                (foreheadToFaceRatio > 0.9 ? (foreheadToFaceRatio - 0.9) * 600 : 0) + 60,
                100
            );

            // 8. Masculinity Score - based on jaw prominence and face structure
            const masculinityScore = Math.min(
                (jawlineScore * 0.4) + (cheekboneScore * 0.3) + (foreheadScore * 0.3),
                100
            );

            // 9. Skin Quality - analyze from actual image pixels
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Extract skin region (between eyes and mouth)
            const skinRegion = {
                x: Math.min(landmarks[33][0], landmarks[362][0]),
                y: landmarks[168][1], // nose bridge
                width: Math.abs(landmarks[362][0] - landmarks[33][0]),
                height: landmarks[13][1] - landmarks[168][1] // to upper lip
            };

            const imageData = ctx.getImageData(
                skinRegion.x,
                skinRegion.y,
                skinRegion.width,
                skinRegion.height
            );

            // Calculate skin quality metrics
            let rTotal = 0, gTotal = 0, bTotal = 0;
            let variance = 0;
            const pixels = imageData.data;
            const pixelCount = pixels.length / 4;

            // Calculate average color
            for (let i = 0; i < pixels.length; i += 4) {
                rTotal += pixels[i];
                gTotal += pixels[i + 1];
                bTotal += pixels[i + 2];
            }

            const avgR = rTotal / pixelCount;
            const avgG = gTotal / pixelCount;
            const avgB = bTotal / pixelCount;

            // Calculate variance (smoothness indicator)
            for (let i = 0; i < pixels.length; i += 4) {
                const rDiff = pixels[i] - avgR;
                const gDiff = pixels[i + 1] - avgG;
                const bDiff = pixels[i + 2] - avgB;
                variance += (rDiff * rDiff + gDiff * gDiff + bDiff * bDiff) / 3;
            }

            variance = Math.sqrt(variance / pixelCount);

            // Lower variance = smoother skin = higher score
            // Also consider brightness (avoiding too dark/light)
            const brightness = (avgR + avgG + avgB) / 3;
            const brightnessScore = Math.max(0, 100 - Math.abs(brightness - 140) * 0.5);
            const smoothnessScore = Math.max(0, 100 - variance * 0.8);
            const skinQualityScore = (smoothnessScore * 0.6 + brightnessScore * 0.4);

            // 10. Hairstyle - analyze forehead coverage and hair presence
            const hairRegion = {
                x: landmarks[109][0],
                y: Math.max(0, landmarks[10][1] - (faceHeight * 0.3)),
                width: landmarks[338][0] - landmarks[109][0],
                height: landmarks[10][1] - Math.max(0, landmarks[10][1] - (faceHeight * 0.3))
            };

            const hairData = ctx.getImageData(
                hairRegion.x,
                hairRegion.y,
                hairRegion.width,
                hairRegion.height
            );

            // Analyze hair region darkness and contrast
            let hairDarkness = 0;
            let hairVariance = 0;
            const hairPixels = hairData.data;
            const hairPixelCount = hairPixels.length / 4;

            for (let i = 0; i < hairPixels.length; i += 4) {
                const gray = (hairPixels[i] + hairPixels[i + 1] + hairPixels[i + 2]) / 3;
                hairDarkness += gray;
            }

            hairDarkness = hairDarkness / hairPixelCount;

            // Calculate hair contrast (indicates defined hair vs bald)
            for (let i = 0; i < hairPixels.length; i += 4) {
                const gray = (hairPixels[i] + hairPixels[i + 1] + hairPixels[i + 2]) / 3;
                hairVariance += Math.pow(gray - hairDarkness, 2);
            }

            hairVariance = Math.sqrt(hairVariance / hairPixelCount);

            // Good hair: decent contrast, not too bright (bald)
            const hairPresenceScore = hairDarkness < 180 ? 70 : 40; // Dark hair present
            const hairContrastScore = Math.min(hairVariance * 0.5, 30); // Texture/definition
            const hairstyleScore = Math.min(hairPresenceScore + hairContrastScore, 100);

            // Calculate weighted overall score
            const weights = {
                symmetry: 0.20,
                jawline: 0.15,
                cheekbones: 0.15,
                nose: 0.12,
                eyeArea: 0.12,
                faceShape: 0.10,
                forehead: 0.08,
                masculinity: 0.08
            };

            const currentScore = Math.round(
                finalSymmetry * weights.symmetry +
                jawlineScore * weights.jawline +
                cheekboneScore * weights.cheekbones +
                noseScore * weights.nose +
                eyeScore * weights.eyeArea +
                ratioScore * weights.faceShape +
                foreheadScore * weights.forehead +
                masculinityScore * weights.masculinity
            );

            // Potential score calculation
            const improvableScores = {
                skinQuality: Math.min(skinQualityScore + 15, 95),
                hairstyle: Math.min(hairstyleScore + 20, 95),
                jawline: Math.min(jawlineScore + 12, 95),
                cheekbones: Math.min(cheekboneScore + 10, 95)
            };

            const potentialScore = Math.min(currentScore + 15, 95);

            setLoading(false);

            return {
                current: currentScore,
                potential: potentialScore,
                features: {
                    symmetry: Math.round(finalSymmetry),
                    jawline: Math.round(jawlineScore),
                    cheekbones: Math.round(cheekboneScore),
                    noseShape: Math.round(noseScore),
                    eyeArea: Math.round(eyeScore),
                    faceShape: Math.round(ratioScore),
                    forehead: Math.round(foreheadScore),
                    masculinity: Math.round(masculinityScore),
                    skinQuality: Math.round(skinQualityScore),
                    hairstyle: Math.round(hairstyleScore)
                },
                measurements: {
                    faceShape: calculateFaceShape(faceRatio, jawToFaceRatio, cheekboneToFaceRatio, landmarks),
                    eyeType: calculateEyeType(eyeToFaceRatio, leftEyeWidth, rightEyeWidth, eyeDistance, landmarks),
                    canthalTilt: calculateCanthalTilt(landmarks),
                    maxillaDevelopment: calculateMaxillaDevelopment(cheekboneToFaceRatio, landmarks),
                    noseShape: calculateNoseShape(landmarks, noseToFaceRatio, faceRatio)
                }
            };

        } catch (error) {
            console.error('Error analyzing face:', error);
            setLoading(false);
            alert('Error analyzing face. Please try another photo.');
            return null;
        }
    };

    const processFile = async (file) => {
        if (!file) return;
        
        // Check file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, etc.)');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File too large. Please select an image smaller than 10MB.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = async (event) => {
            setImage(event.target.result);
            const analysis = await analyzeFacialFeatures(event.target.result);
            if (analysis) {
                setScores(analysis);
                setStep(2);
            }
        };
        reader.readAsDataURL(file);
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        await processFile(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragOver(false);
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        setDragOver(false);
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            await processFile(files[0]);
        }
    };

    const checkCameraPermission = async () => {
        try {
            const permission = await navigator.permissions.query({ name: 'camera' });
            return permission.state;
        } catch (error) {
            // Permissions API not supported, continue with getUserMedia
            return 'unknown';
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => {
                track.stop();
            });
            setStream(null);
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setCameraStatus('idle');
    };

    const startCamera = async () => {
        try {
            console.log('=== STARTING CAMERA ===');
            setCameraStatus('starting');
            
            // Check getUserMedia support
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('Camera is not supported by your browser');
            }

            // Request camera with simpler constraints
            console.log('Requesting camera access...');
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            });
            
            console.log('✅ Got camera stream:', stream);
            
            // Set stream first to make video element appear
            setStream(stream);
            
            // Wait for React to render the video element
            await new Promise(resolve => setTimeout(resolve, 100));
            
            if (!videoRef.current) {
                throw new Error('Video element still not found after stream set');
            }

            // Set up video element
            const video = videoRef.current;
            video.srcObject = stream;
            
            // Use Promise approach for video loading
            const initVideo = () => {
                return new Promise((resolve, reject) => {
                    const checkVideo = () => {
                        if (video.videoWidth > 0 && video.videoHeight > 0) {
                            console.log('✅ Camera ready:', video.videoWidth, 'x', video.videoHeight);
                            resolve();
                        } else {
                            console.log('Waiting for video dimensions...');
                            setTimeout(checkVideo, 100);
                        }
                    };
                    
                    video.onloadedmetadata = () => {
                        console.log('📹 Metadata loaded, playing video...');
                        video.play()
                            .then(checkVideo)
                            .catch(reject);
                    };
                    
                    video.onerror = reject;
                    
                    // Timeout after 10 seconds
                    setTimeout(() => reject(new Error('Video initialization timeout')), 10000);
                });
            };

            await initVideo();
            setCameraStatus('ready');
            
        } catch (err) {
            console.error('❌ Camera error:', err);
            setCameraStatus('error');
            
            // Clean up stream if error occurred
            if (stream && stream.getTracks) {
                stream.getTracks().forEach(track => track.stop());
            }
            setStream(null);
            
            // Show user-friendly error message
            setTimeout(() => {
                const message = err.name === 'NotAllowedError' 
                    ? 'Camera access denied.'
                    : err.name === 'NotFoundError'
                    ? 'No camera found.'
                    : `Camera error: ${err.message}`;
                    
                alert(message + ' Please use the "Upload Photo" option instead.');
            }, 100);
        }
    };

    const capturePhoto = async () => {
        const video = videoRef.current;
        
        if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
            alert('Camera not ready. Please wait a moment and try again.');
            return;
        }

        try {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const photo = canvas.toDataURL('image/jpeg', 0.9);
            setImage(photo);

            // Stop camera stream
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                setStream(null);
            }

            // Analyze the captured photo
            const analysis = await analyzeFacialFeatures(photo);
            if (analysis) {
                setScores(analysis);
                setStep(2);
            }
        } catch (error) {
            console.error('Photo capture error:', error);
            alert('Failed to capture photo. Please try again or use file upload.');
        }
    };

    // Generate detailed AI-powered guide for a specific recommendation
    const getDetailedAIGuide = async (featureName, recommendation, score) => {
        console.log(`📚 Generating detailed guide for ${featureName}: ${recommendation}`);
        
        const prompt = `You are a professional facial aesthetics expert. Generate a comprehensive, detailed guide for this recommendation:

Feature: ${featureName}
Current Score: ${score}/100
Recommendation: ${recommendation}

Provide a detailed guide in JSON format with the following structure:
{
  "title": "Clear, engaging title",
  "overview": "2-3 sentence overview explaining the technique and its benefits",
  "steps": [
    {
      "stepNumber": 1,
      "title": "Step title",
      "description": "Detailed description",
      "duration": "How long this step takes",
      "frequency": "How often to do this",
      "tips": ["Tip 1", "Tip 2"]
    }
  ],
  "expectedResults": {
    "week1": "What to expect in week 1",
    "week4": "What to expect in week 4",
    "week8": "What to expect in week 8"
  },
  "dos": ["Do this", "Do that"],
  "donts": ["Don't do this", "Don't do that"],
  "proTips": ["Expert tip 1", "Expert tip 2"],
  "relatedProducts": ["Product recommendation 1", "Product recommendation 2"]
}

Provide specific, actionable, and realistic advice. Return only valid JSON, no markdown or extra text.`;

        const providers = [
            {
                name: 'Groq',
                url: 'https://api.groq.com/openai/v1/chat/completions',
                key: import.meta.env.VITE_GROQ_API_KEY,
                model: 'llama-3.1-8b-instant'
            },
            {
                name: 'OpenAI',
                url: 'https://api.openai.com/v1/chat/completions',
                key: import.meta.env.VITE_OPENAI_API_KEY,
                model: 'gpt-3.5-turbo'
            }
        ];

        for (const provider of providers) {
            if (!provider.key) continue;
            
            try {
                console.log(`Trying ${provider.name} for detailed guide...`);
                const response = await fetch(provider.url, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${provider.key}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: provider.model,
                        messages: [{ role: 'user', content: prompt }],
                        temperature: 0.7,
                        max_tokens: 2000,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`${provider.name} API error: ${response.status}`);
                }

                const data = await response.json();
                const content = data.choices[0].message.content.trim();
                
                // Try to parse JSON, removing markdown code blocks if present
                const jsonMatch = content.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    const guide = JSON.parse(jsonMatch[0]);
                    console.log(`Successfully got detailed guide from ${provider.name}`);
                    return guide;
                }
            } catch (error) {
                console.error(`${provider.name} failed:`, error.message);
                continue;
            }
        }

        // Fallback guide if AI fails
        return {
            title: "Comprehensive Improvement Guide",
            overview: "This guide provides evidence-based techniques to improve your facial aesthetics through consistent practice and proper care.",
            steps: [
                {
                    stepNumber: 1,
                    title: "Start Your Routine",
                    description: "Begin with a clean, relaxed face. Take before photos to track progress.",
                    duration: "5 minutes",
                    frequency: "Daily",
                    tips: ["Use good lighting for photos", "Relax facial muscles first"]
                },
                {
                    stepNumber: 2,
                    title: "Apply the Technique",
                    description: recommendation,
                    duration: "10-15 minutes",
                    frequency: "Twice daily",
                    tips: ["Be consistent", "Don't overdo it", "Listen to your body"]
                },
                {
                    stepNumber: 3,
                    title: "Track and Adjust",
                    description: "Monitor your progress weekly and adjust intensity as needed.",
                    duration: "5 minutes weekly",
                    frequency: "Weekly",
                    tips: ["Take weekly progress photos", "Note any improvements", "Stay patient"]
                }
            ],
            expectedResults: {
                week1: "Initial adaptation phase - you may feel slight muscle activation",
                week4: "Noticeable improvements in muscle tone and definition",
                week8: "Visible enhancement in facial structure and symmetry"
            },
            dos: [
                "Be consistent with your routine",
                "Track progress with photos",
                "Stay hydrated and maintain good sleep",
                "Combine with healthy lifestyle habits"
            ],
            donts: [
                "Don't expect overnight results",
                "Don't skip days frequently",
                "Don't overdo exercises or products",
                "Don't compare yourself to others"
            ],
            proTips: [
                "Set daily reminders for consistency",
                "Join online communities for motivation",
                "Pair with proper nutrition and hydration",
                "Results typically visible in 4-8 weeks"
            ],
            relatedProducts: [
                "Jade roller for facial massage",
                "Quality moisturizer for skin health",
                "Vitamin C serum for brightness"
            ]
        };
    };

    // Multi-provider AI Recommendations (Groq, OpenAI, Hugging Face)
    const getAIRecommendations = async (featureName, score, facialMeasurements) => {
        console.log(`🚀 Starting AI recommendation for ${featureName}`);
        
        const prompt = `You are a professional facial aesthetics coach and beauty expert. 
        
        Analyze this facial feature data:
        - Feature: ${featureName}
        - Current Score: ${score}/100
        - Facial Measurements: ${JSON.stringify(facialMeasurements)}
        
        Provide 4-6 specific, actionable recommendations to improve the ${featureName} feature and overall facial attractiveness. Focus on:
        1. Natural improvement methods (exercises, skincare, lifestyle)
        2. Professional treatments (if applicable)
        3. Styling tips and techniques
        4. Preventive measures
        
        Format as a JSON array of strings. Each recommendation should be concise (max 80 characters) and actionable.
        Example: ["Use retinol serum nightly", "Practice jaw exercises daily", "Get 8+ hours sleep"]
        
        Return only the JSON array, no other text.`;

        // Try multiple providers in order of preference
        const providers = [
            {
                name: 'Groq (Free & Fast)',
                url: 'https://api.groq.com/openai/v1/chat/completions',
                key: import.meta.env.VITE_GROQ_API_KEY,
                model: 'llama-3.1-8b-instant'
            },
            {
                name: 'OpenAI',
                url: 'https://api.openai.com/v1/chat/completions',
                key: import.meta.env.VITE_OPENAI_API_KEY,
                model: 'gpt-3.5-turbo'
            },
            {
                name: 'Hugging Face',
                url: 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-large',
                key: import.meta.env.VITE_HF_API_KEY,
                model: 'microsoft/DialoGPT-large'
            }
        ];

        for (const provider of providers) {
            if (!provider.key || provider.key === 'your-api-key-here') continue;
            
            try {
                console.log(`📤 Trying ${provider.name}...`);
                
                const response = await fetch(provider.url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + provider.key
                    },
                    body: JSON.stringify({
                        model: provider.model,
                        messages: [{
                            role: 'system',
                            content: 'You are an expert facial aesthetics coach. Always respond with valid JSON arrays only.'
                        }, {
                            role: 'user',
                            content: prompt
                        }],
                        max_tokens: 300,
                        temperature: 0.7
                    })
                });

                console.log('📥 Response status:', response.status);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log(`✅ ${provider.name} success:`, data);
                    const content = data.choices[0].message.content.trim();
                    
                    try {
                        const parsedRecommendations = JSON.parse(content);
                        console.log(`🎉 Got recommendations from ${provider.name}:`, parsedRecommendations);
                        return parsedRecommendations;
                    } catch (parseError) {
                        console.warn(`❌ Failed to parse ${provider.name} response, trying next provider...`);
                        continue; // Try next provider
                    }
                } else {
                    console.warn(`❌ ${provider.name} failed with status:`, response.status);
                    continue; // Try next provider
                }
                
            } catch (error) {
                console.warn(`❌ ${provider.name} error:`, error.message);
                continue; // Try next provider
            }
        }
        
        // If all providers failed, use enhanced fallback
        console.log('🔄 All AI providers failed, using enhanced recommendations');
        return getFallbackRecommendations(featureName, score);
    };

    const getFallbackRecommendations = (feature, score) => {
        // Enhanced AI-like recommendations based on score ranges
        const recs = {
            symmetry: {
                veryLow: ['Start facial massage therapy 2x daily', 'Sleep exclusively on your back', 'Practice mirror symmetry exercises', 'Chew food alternating sides equally', 'Consider facial cupping therapy'],
                low: ['Practice facial exercises evenly on both sides', 'Sleep on your back to avoid facial pressure', 'Chew food evenly on both sides', 'Facial massage for muscle balance', 'Use jade roller for lymphatic drainage'],
                medium: ['Maintain current symmetry routine', 'Weekly professional facial massage', 'Continue balanced chewing habits']
            },
            jawline: {
                veryLow: ['Mewing technique 4+ hours daily', 'Chew jaw trainer gum 45min/day', 'Reduce body fat to 10-15%', 'Testosterone optimization protocol', 'Consider jawline filler consultation'],
                low: ['Jaw exercises: chin lifts daily', 'Chew harder foods (carrots, nuts)', 'Mewing technique: tongue on palate', 'Reduce sodium to minimize bloating', 'Maintain low body fat (12-18%)'],
                medium: ['Continue mewing practice', 'Maintain lean physique', 'Weekly jawline massage']
            },
            cheekbones: {
                veryLow: ['Buccal fat reduction consultation', 'Intensive face yoga 2x daily', 'Achieve 8-12% body fat', 'Cheekbone contouring makeup daily', 'Consider dermal filler enhancement'],
                low: ['Face yoga: cheek lifts', 'Chew sugar-free gum 30min daily', 'Facial massage upward strokes', 'Reduce overall body fat', 'Gua sha stone therapy'],
                medium: ['Maintain facial exercises', 'Continue body fat optimization', 'Weekly gua sha treatment']
            },
            noseShape: {
                veryLow: ['Non-surgical nose job consultation', 'Daily nose contouring makeup', 'Nose exercise routine 3x daily', 'Breathing exercises for nose function', 'Consider rhinoplasty consultation'],
                low: ['Nose exercises daily', 'Proper breathing through nose', 'Facial contouring techniques', 'Consider professional consultation', 'Nose massage with oils'],
                medium: ['Continue nose exercises', 'Maintain breathing practices', 'Subtle contouring when needed']
            },
            eyeArea: {
                veryLow: ['Under-eye filler consultation', 'Professional LED light therapy', 'Prescription retinoid treatment', 'Botox for crow\'s feet', 'Vitamin C + E serum combo'],
                low: ['Eye cream with caffeine & retinol', 'Get 7-8 hours sleep', 'Reduce screen time, use blue light filter', 'Stay hydrated (8+ glasses)', 'Cold compress morning'],
                medium: ['Maintain eye care routine', 'Continue sleep optimization', 'Weekly eye masks']
            },
            faceShape: {
                veryLow: ['Surgical consultation for face shape', 'Strategic beard styling', 'Professional hairstyle consultation', 'Face slimming exercises daily', 'Optimize facial hair growth'],
                low: ['Overall facial exercises', 'Maintain healthy weight', 'Proper posture', 'Strategic hairstyle choice', 'Face yoga comprehensive routine'],
                medium: ['Continue face exercises', 'Maintain ideal weight', 'Regular hairstyle updates']
            },
            forehead: {
                veryLow: ['Botox for forehead lines', 'Prescription retinoid cream', 'LED light therapy sessions', 'Micro-needling treatment', 'Bangs hairstyle consideration'],
                low: ['Forehead massage', 'Reduce frowning', 'Botox for lines (optional)', 'Hairstyle to complement', 'Daily SPF 50+ protection'],
                medium: ['Maintain skincare routine', 'Continue sun protection', 'Occasional professional treatments']
            },
            masculinity: {
                veryLow: ['Testosterone replacement therapy consult', 'Heavy compound lifting 4x/week', 'Grow full beard if possible', 'Posture coaching sessions', 'Voice deepening exercises'],
                low: ['Strength training compound lifts', 'Grow facial hair if possible', 'Improve posture', 'Dress in structured clothing', 'Increase testosterone naturally'],
                medium: ['Maintain fitness routine', 'Continue grooming practices', 'Optimize hormone levels']
            },
            skinQuality: {
                veryLow: ['Dermatologist consultation ASAP', 'Professional chemical peel series', 'Prescription tretinoin 0.05%', 'LED therapy + microneedling', 'Comprehensive supplement stack'],
                low: ['Daily cleansing routine 2x', 'Vitamin C serum (morning)', 'Retinol serum (night)', 'Sunscreen SPF 50+ daily', 'Hydrate well + 7-8 hours sleep'],
                medium: ['Maintain skincare regimen', 'Monthly professional facial', 'Continue sun protection']
            },
            hairstyle: {
                veryLow: ['Hair transplant consultation', 'Professional stylist monthly', 'Hair growth treatment protocol', 'Modern trendy cut update', 'Quality styling product investment'],
                low: ['Consult professional stylist', 'Modern textured cut', 'Use quality styling products', 'Consider hair growth supplements', 'Update cut every 6-8 weeks'],
                medium: ['Maintain current style', 'Regular trims', 'Quality product maintenance']
            }
        };

        // Determine recommendation level based on score
        if (score < 50) return recs[feature]?.veryLow || ['Consult professionals for improvement options'];
        if (score < 70) return recs[feature]?.low || ['Focus on improvement techniques'];
        if (score < 85) return recs[feature]?.medium || ['Maintain and optimize current routine'];
        return ['Excellent! Focus on maintenance and fine-tuning', 'Consider helping others with your routine'];
    };

    // Generate dynamic action plan based on AI recommendations and scores
    const generateActionPlan = (aiRecommendations, scores) => {
        if (!aiRecommendations || Object.keys(aiRecommendations).length === 0) {
            return {
                week1: "Start skincare routine, begin facial exercises",
                week2: "Add mewing practice, optimize diet & hydration", 
                month1: "Consistent exercise, consider hairstyle change",
                month2: "Review progress, adjust routine, maintain gains"
            };
        }

        const allRecommendations = Object.values(aiRecommendations).flat();
        const lowScores = Object.entries(scores.features).filter(([_, score]) => score < 70);
        const totalRecommendations = allRecommendations.length;
        
        // Categorize recommendations by type
        const skincare = allRecommendations.filter(rec => 
            rec.toLowerCase().includes('serum') || 
            rec.toLowerCase().includes('cream') || 
            rec.toLowerCase().includes('skincare') ||
            rec.toLowerCase().includes('spf') ||
            rec.toLowerCase().includes('retinol')
        );
        
        const exercises = allRecommendations.filter(rec => 
            rec.toLowerCase().includes('exercise') || 
            rec.toLowerCase().includes('mewing') || 
            rec.toLowerCase().includes('massage') ||
            rec.toLowerCase().includes('yoga') ||
            rec.toLowerCase().includes('lift')
        );
        
        const lifestyle = allRecommendations.filter(rec => 
            rec.toLowerCase().includes('sleep') || 
            rec.toLowerCase().includes('hydrat') || 
            rec.toLowerCase().includes('diet') ||
            rec.toLowerCase().includes('posture') ||
            rec.toLowerCase().includes('stress')
        );
        
        const professional = allRecommendations.filter(rec => 
            rec.toLowerCase().includes('consult') || 
            rec.toLowerCase().includes('professional') || 
            rec.toLowerCase().includes('dermat') ||
            rec.toLowerCase().includes('stylist') ||
            rec.toLowerCase().includes('botox') ||
            rec.toLowerCase().includes('filler')
        );

        // Generate timeline based on severity and number of recommendations
        const severity = lowScores.length > 4 ? 'high' : lowScores.length > 2 ? 'medium' : 'low';
        const intensity = totalRecommendations > 20 ? 'intensive' : totalRecommendations > 10 ? 'moderate' : 'light';
        
        if (severity === 'high' || intensity === 'intensive') {
            return {
                week1: `Start ${skincare.length > 0 ? 'daily skincare routine' : 'basic care'} + ${exercises.length > 0 ? 'begin facial exercises' : 'posture work'}`,
                week2: `Add ${lifestyle.length > 0 ? lifestyle[0]?.toLowerCase() : 'hydration focus'} + continue exercises`,
                month1: `${exercises.length > 1 ? 'Advanced exercises' : 'Consistent routine'} + ${professional.length > 0 ? 'schedule consultations' : 'optimize diet'}`,
                month2: `${professional.length > 0 ? 'Professional treatments' : 'Maintain routine'} + track progress daily`
            };
        } else if (severity === 'medium') {
            return {
                week1: `Begin ${skincare.length > 0 ? skincare[0]?.toLowerCase() : 'skincare basics'}`,
                week2: `Add ${exercises.length > 0 ? 'facial exercises' : 'posture improvement'}`,
                month1: `${lifestyle.length > 0 ? lifestyle[0]?.toLowerCase() : 'Lifestyle optimization'} + consistent routine`,
                month2: `Fine-tune routine + ${professional.length > 0 ? 'consider professional advice' : 'maintain progress'}`
            };
        } else {
            return {
                week1: `Start gentle improvements: ${skincare[0]?.toLowerCase() || 'basic skincare'}`,
                week2: `Add: ${exercises[0]?.toLowerCase() || 'light exercises'}`,
                month1: `Maintain routine + ${lifestyle[0]?.toLowerCase() || 'optimize sleep'}`,
                month2: `Perfect routine + enjoy your progress!`
            };
        }
    };

    const ScoreCard = ({ label, score }) => (
        <div className="relative group">
            <div className="relative overflow-hidden rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 p-6 hover:border-teal-500/30 hover:shadow-2xl hover:shadow-teal-500/20 transition-all duration-500 cursor-pointer hover:scale-105">
                <div className="absolute inset-0 bg-gradient-to-br from-teal-500/0 to-cyan-500/0 group-hover:from-teal-500/10 group-hover:to-cyan-500/10 transition-all duration-500"></div>
                
                <div className="relative text-center">
                    <h3 className="text-white/60 text-xs font-bold mb-4 uppercase tracking-widest">{label}</h3>
                    <div className="text-5xl font-bold text-white mb-4">{score}</div>
                    <div className="w-full bg-white/10 rounded-full h-2 mb-3 overflow-hidden">
                        <div
                            className={`h-2 rounded-full transition-all duration-700 ${
                                score >= 80 ? 'bg-gradient-to-r from-green-400 to-emerald-500' :
                                score >= 60 ? 'bg-gradient-to-r from-yellow-400 to-orange-500' :
                                'bg-gradient-to-r from-pink-400 to-rose-500'
                            }`}
                            style={{ width: `${score}%` }}
                        />
                    </div>
                    <div className={`text-xs font-bold uppercase tracking-wider ${
                        score >= 80 ? 'text-green-400' :
                        score >= 60 ? 'text-yellow-400' :
                        'text-pink-400'
                    }`}>
                        {score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Improve'}
                    </div>
                </div>
            </div>
        </div>
    );

    return (
        <div className={`min-h-screen bg-gradient-to-br ${currentTheme.bg} relative overflow-hidden transition-colors duration-1000`}>
            {console.log('App render - step:', step, 'scores:', !!scores, 'loadingRecommendations:', loadingRecommendations)}
            
            {/* Theme Selector Button */}
            <div className="fixed top-6 right-6 z-50">
                <button
                    onClick={() => setShowThemeSelector(!showThemeSelector)}
                    className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-xl border border-white/20 hover:bg-white/20 transition-all duration-300 flex items-center justify-center shadow-2xl group"
                    title="Change Theme"
                >
                    <div className={`w-8 h-8 rounded-full bg-gradient-to-r ${currentTheme.primary} group-hover:scale-110 transition-transform duration-300`}></div>
                </button>
                
                {/* Theme Selector Panel */}
                {showThemeSelector && (
                    <div className="absolute top-16 right-0 bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-4 shadow-2xl animate-fadeInUp min-w-[280px]">
                        <h3 className="text-white font-bold mb-4 px-2">Choose Theme</h3>
                        <div className="space-y-2">
                            {Object.entries(themes).map(([key, themeConfig]) => (
                                <button
                                    key={key}
                                    onClick={() => {
                                        setTheme(key);
                                        setShowThemeSelector(false);
                                    }}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all duration-300 ${
                                        theme === key 
                                            ? 'bg-white/20 border-2 border-white/40' 
                                            : 'bg-white/5 border border-white/10 hover:bg-white/10'
                                    }`}
                                >
                                    <div className={`w-10 h-10 rounded-xl bg-gradient-to-r ${themeConfig.primary} flex-shrink-0`}></div>
                                    <div className="text-left flex-1">
                                        <p className="text-white font-semibold text-sm">{themeConfig.name}</p>
                                        {theme === key && <p className="text-white/60 text-xs">Active</p>}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>
            
            {/* Dynamic animated background pattern */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                {/* Animated gradient orbs */}
                <div className={`absolute top-0 -left-4 w-96 h-96 ${currentTheme.orb1} rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-move-bg transition-colors duration-1000`}></div>
                <div className={`absolute top-0 -right-4 w-96 h-96 ${currentTheme.orb2} rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-move-bg transition-colors duration-1000`} style={{animationDelay: '7s', animationDuration: '25s'}}></div>
                <div className={`absolute -bottom-8 left-20 w-96 h-96 ${currentTheme.orb3} rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-move-bg transition-colors duration-1000`} style={{animationDelay: '14s', animationDuration: '30s'}}></div>
                
                {/* Rotating gradient rings */}
                <div className={`absolute top-1/4 left-1/4 w-[600px] h-[600px] border border-${currentTheme.accent}/10 rounded-full animate-rotate-slow transition-colors duration-1000`}></div>
                <div className={`absolute top-1/3 right-1/4 w-[500px] h-[500px] border border-${currentTheme.accent}/10 rounded-full animate-rotate-slow transition-colors duration-1000`} style={{animationDuration: '40s', animationDirection: 'reverse'}}></div>
                
                {/* Grid pattern overlay - using inline style for dynamic color */}
                <div className="absolute inset-0 opacity-50" style={{
                    backgroundImage: `linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)`,
                    backgroundSize: '50px 50px',
                    color: `rgba(${theme === 'teal' ? '6,182,212' : theme === 'purple' ? '168,85,247' : theme === 'blue' ? '59,130,246' : theme === 'green' ? '16,185,129' : '244,63,94'}, 0.03)`
                }}></div>
            </div>

            <div className="relative z-10 p-4 md:p-8">
                <div className="max-w-7xl mx-auto">
                    {/* Premium Header */}
                    <div className="text-center mb-16 animate-fadeInUp">
                        <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md rounded-full px-4 py-2 mb-6 border border-white/20">
                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                            {modelLoading ? (
                                <>
                                    <Loader className="w-4 h-4 text-white/80 animate-spin" />
                                    <span className="text-white/80 text-sm font-medium">Initializing AI...</span>
                                </>
                            ) : model ? (
                                <>
                                    <span className="text-green-400 text-sm">●</span>
                                    <span className="text-white/90 text-sm font-medium">AI Powered</span>
                                </>
                            ) : (
                                <span className="text-white/70 text-sm font-medium">Loading...</span>
                            )}
                        </div>
                        
                        <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-4 tracking-tight">
                            <span className={`bg-gradient-to-r ${currentTheme.text} bg-clip-text text-transparent transition-all duration-1000`}>
                                FaceMaxx
                            </span>
                        </h1>
                        <p className="text-xl md:text-2xl text-white/70 font-light max-w-2xl mx-auto leading-relaxed">
                            Premium AI-powered facial analysis with personalized recommendations
                        </p>
                        
                        {modelError && (
                            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                                <p className="text-red-700 text-sm">{modelError}</p>
                                <div className="flex gap-2 mt-2">
                                    <button 
                                        onClick={() => {
                                            setModel(null);
                                            setModelError(null);
                                            loadModel();
                                        }}
                                        className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
                                    >
                                        Retry Loading Model
                                    </button>
                                    <button 
                                        onClick={() => setModelError(null)}
                                        className="px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-700"
                                    >
                                        Continue Without AI
                                    </button>
                                    <button 
                                        onClick={() => window.location.reload()}
                                        className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                                    >
                                        Refresh Page
                                    </button>
                                </div>
                                <p className="text-xs text-gray-600 mt-2">
                                    You can still upload photos and get basic analysis without the AI model
                                </p>
                            </div>
                        )}
                    </div>

                    {loading && (
                        <div className="fixed inset-0 bg-black/70 backdrop-blur-2xl flex items-center justify-center z-50 animate-fadeInUp">
                            <div className="relative">
                                {/* Animated gradient blob background */}
                                <div className="absolute inset-0 -z-10">
                                    <div className="absolute top-0 left-0 w-72 h-72 bg-teal-500/30 rounded-full blur-3xl animate-pulse"></div>
                                    <div className="absolute bottom-0 right-0 w-72 h-72 bg-orange-500/30 rounded-full blur-3xl animate-pulse" style={{animationDelay: '0.5s'}}></div>
                                </div>
                                
                                <div className="relative bg-white/10 backdrop-blur-xl border border-white/20 p-12 rounded-3xl text-center max-w-md mx-4 shadow-2xl">
                                    {/* Premium loader */}
                                    <div className="relative w-24 h-24 mx-auto mb-8">
                                        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 animate-spin" style={{clipPath: 'polygon(50% 50%, 100% 0, 100% 100%)'}}></div>
                                        <div className="absolute inset-2 rounded-full bg-gradient-to-br from-slate-950 to-teal-950 flex items-center justify-center">
                                            <Loader className="w-10 h-10 text-white animate-spin" />
                                        </div>
                                    </div>
                                    
                                    <h3 className="text-3xl font-bold text-white mb-4 bg-gradient-to-r from-white via-teal-200 to-cyan-200 bg-clip-text text-transparent">
                                        {modelLoading ? 'Loading AI Model' : 'Analyzing Your Features'}
                                    </h3>
                                    
                                    <p className="text-white/70 text-base leading-relaxed mb-4">
                                        {modelLoading 
                                            ? 'Preparing TensorFlow.js and face detection model' 
                                            : model 
                                                ? 'Using AI Model for high-precision analysis'
                                                : 'Using Basic Analysis mode (AI model unavailable)'
                                        }
                                    </p>
                                    
                                    {modelLoading && (
                                        <div className="mt-6 p-4 bg-white/5 rounded-2xl border border-white/10">
                                            <p className="text-sm text-white/60">This may take 10-30 seconds on first load</p>
                                        </div>
                                    )}
                                    
                                    {!model && !modelLoading && (
                                        <div className="mt-6 p-4 bg-orange-500/10 rounded-2xl border border-orange-500/30">
                                            <p className="text-sm text-orange-300">Using estimated analysis - results may be less accurate</p>
                                        </div>
                                    )}
                                    
                                    {/* Progress dots */}
                                    <div className="mt-8 flex justify-center gap-2">
                                        <div className="w-2 h-2 bg-teal-400 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                                        <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {step === 1 && (
                        <div className="space-y-8 animate-fadeInUp">
                            {!stream ? (
                                <div className="grid md:grid-cols-2 gap-6">
                                    {/* Premium File Upload Card */}
                                    <div
                                        onClick={() => fileInputRef.current.click()}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                        onDrop={handleDrop}
                                        className={`relative group cursor-pointer transition-all duration-500 ${
                                            dragOver ? 'scale-105' : 'hover:scale-102'
                                        }`}
                                    >
                                        <div className={`relative overflow-hidden rounded-3xl p-10 h-full backdrop-blur-xl border transition-all duration-500 ${
                                            dragOver 
                                                ? 'bg-gradient-to-br from-pink-500/20 to-purple-500/20 border-pink-400/50 shadow-2xl shadow-pink-500/30' 
                                                : 'bg-white/10 border-white/20 shadow-xl hover:shadow-2xl hover:border-white/30'
                                        }`}>
                                            {/* Animated gradient overlay */}
                                            <div className="absolute inset-0 bg-gradient-to-br from-pink-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                            
                                            {/* Recommended badge */}
                                            <div className={`absolute top-6 right-6 bg-gradient-to-r ${currentTheme.primary} text-white px-4 py-2 rounded-full text-xs font-semibold shadow-lg ${currentTheme.glow} animate-glow transition-all duration-1000`}>
                                                ✨ Recommended
                                            </div>
                                            
                                            <div className="relative z-10 flex flex-col items-center text-center h-full justify-center">
                                                <div className={`w-24 h-24 rounded-2xl bg-gradient-to-br ${currentTheme.primary} flex items-center justify-center mb-6 shadow-lg transition-all duration-700 ${
                                                    dragOver ? 'scale-110 rotate-12' : 'group-hover:scale-110'
                                                }`}>
                                                    <Upload className="w-12 h-12 text-white" />
                                                </div>
                                                
                                                <h3 className="text-2xl font-bold text-white mb-3">
                                                    Upload Photo
                                                </h3>
                                                <p className="text-white/70 text-base leading-relaxed mb-4 max-w-xs">
                                                    {dragOver 
                                                        ? '✨ Drop your photo here' 
                                                        : 'Click or drag & drop your best photo'
                                                    }
                                                </p>
                                                <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-full px-4 py-2 text-sm text-white/80 border border-white/20">
                                                    <span>JPG, PNG</span>
                                                    <span className="w-1 h-1 bg-white/40 rounded-full"></span>
                                                    <span>Max 10MB</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Premium Camera Card */}
                                    <div
                                        onClick={async () => {
                                            console.log('Camera button clicked');
                                            try {
                                                const testStream = await navigator.mediaDevices.getUserMedia({ video: true });
                                                console.log('Quick camera test passed');
                                                testStream.getTracks().forEach(track => track.stop());
                                            } catch (e) {
                                                console.error('Quick camera test failed:', e);
                                                alert('Camera test failed: ' + e.message);
                                                return;
                                            }
                                            setCameraStatus('idle');
                                            startCamera();
                                        }}
                                        className="relative group cursor-pointer transition-all duration-500 hover:scale-102"
                                    >
                                        <div className="relative overflow-hidden rounded-3xl p-10 h-full bg-white/10 backdrop-blur-xl border border-white/20 shadow-xl hover:shadow-2xl hover:border-white/30 transition-all duration-500">
                                            {/* Animated gradient overlay */}
                                            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-coral-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                            
                                            <div className="relative z-10 flex flex-col items-center text-center h-full justify-center">
                                                <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform duration-500">
                                                    <Camera className="w-12 h-12 text-white" />
                                                </div>
                                                
                                                <h3 className="text-2xl font-bold text-white mb-3">
                                                    Use Camera
                                                </h3>
                                                <p className="text-white/70 text-base leading-relaxed mb-4 max-w-xs">
                                                    Take a photo right now with your device camera
                                                </p>
                                                <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-full px-4 py-2 text-sm text-white/80 border border-white/20">
                                                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                                                    <span>Live capture</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-6 animate-fadeInUp">
                                    <div className="relative overflow-hidden rounded-3xl bg-white/5 backdrop-blur-xl border border-white/20 shadow-2xl p-8">
                                        <div className="relative bg-gradient-to-br from-slate-900 to-black rounded-2xl overflow-hidden shadow-inner" style={{minHeight: '400px'}}>
                                            <video 
                                                ref={videoRef} 
                                                autoPlay 
                                                playsInline
                                                muted
                                                className="w-full h-full object-cover rounded-2xl"
                                                style={{minHeight: '400px'}}
                                            />
                                            
                                            {/* Camera frame overlay */}
                                            <div className="absolute inset-0 pointer-events-none">
                                                <div className="absolute inset-8 border-2 border-white/30 rounded-3xl"></div>
                                                <div className="absolute top-8 left-8 w-8 h-8 border-t-4 border-l-4 border-teal-400 rounded-tl-xl"></div>
                                                <div className="absolute top-8 right-8 w-8 h-8 border-t-4 border-r-4 border-teal-400 rounded-tr-xl"></div>
                                                <div className="absolute bottom-8 left-8 w-8 h-8 border-b-4 border-l-4 border-teal-400 rounded-bl-xl"></div>
                                                <div className="absolute bottom-8 right-8 w-8 h-8 border-b-4 border-r-4 border-teal-400 rounded-br-xl"></div>
                                            </div>
                                            
                                            {cameraStatus === 'starting' && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-950/95 to-teal-950/95 backdrop-blur-xl">
                                                    <div className="text-center">
                                                        <div className="w-20 h-20 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 flex items-center justify-center mx-auto mb-6 animate-pulse shadow-lg shadow-teal-500/50">
                                                            <Loader className="w-10 h-10 text-white animate-spin" />
                                                        </div>
                                                        <p className="text-2xl font-bold text-white mb-3">Starting Camera</p>
                                                        <p className="text-white/70">Please grant camera access</p>
                                                        <div className="mt-4 flex justify-center gap-2">
                                                            <div className="w-2 h-2 bg-teal-400 rounded-full animate-bounce"></div>
                                                            <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                                                            <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                            
                                            {cameraStatus === 'error' && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-900/95 to-red-900/95 backdrop-blur-xl">
                                                    <div className="text-center p-8 max-w-md">
                                                        <div className="w-20 h-20 rounded-full bg-red-500/20 flex items-center justify-center mx-auto mb-6 border-2 border-red-500/50">
                                                            <AlertCircle className="w-10 h-10 text-red-400" />
                                                        </div>
                                                        <p className="text-2xl font-bold text-white mb-3">Camera Access Error</p>
                                                        <p className="text-white/70 mb-6">Unable to access your camera. Please check permissions.</p>
                                                        <div className="flex gap-3 justify-center">
                                                            <button 
                                                                onClick={() => {
                                                                    setCameraStatus('idle');
                                                                    startCamera();
                                                                }}
                                                                className="px-6 py-3 bg-gradient-to-r from-teal-500 to-cyan-600 text-white rounded-xl hover:shadow-xl hover:shadow-teal-500/30 transition-all duration-300 font-semibold"
                                                            >
                                                                Try Again
                                                            </button>
                                                            <button 
                                                                onClick={() => {
                                                                    stopCamera();
                                                                }}
                                                                className="px-6 py-3 bg-white/10 backdrop-blur-sm text-white rounded-xl hover:bg-white/20 transition-all duration-300 border border-white/20"
                                                            >
                                                                Cancel
                                                            </button>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        
                                        {cameraStatus === 'ready' && videoRef.current?.videoWidth > 0 && (
                                            <div className="mt-6 flex items-center justify-center gap-3">
                                                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-lg shadow-green-500/50"></div>
                                                <p className="text-white/80 text-sm font-medium">
                                                    Camera Active • {videoRef.current.videoWidth}x{videoRef.current.videoHeight}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                    
                                    <div className="flex gap-4">
                                        <button
                                            onClick={capturePhoto}
                                            disabled={cameraStatus !== 'ready'}
                                            className="flex-1 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            <div className="absolute inset-0 bg-gradient-to-r from-teal-500 to-cyan-600 rounded-2xl group-hover:scale-105 transition-transform duration-300"></div>
                                            <div className="relative z-10 flex items-center justify-center gap-3 py-5 text-white font-bold text-lg">
                                                <Camera className="w-6 h-6" />
                                                <span>Capture & Analyze</span>
                                            </div>
                                        </button>
                                        <button
                                            onClick={() => {
                                                console.log('Stopping camera...');
                                                stopCamera();
                                            }}
                                            className="px-8 bg-white/10 backdrop-blur-sm text-white rounded-2xl hover:bg-white/20 transition-all duration-300 border border-white/20 font-semibold"
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            )}

                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFileUpload}
                                className="hidden"
                            />

                            <div className="mt-8">
                                <div className="relative overflow-hidden rounded-3xl bg-white/5 backdrop-blur-xl border border-white/20 p-8">
                                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-cyan-500/10"></div>
                                    <div className="relative">
                                        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center">
                                                <Camera className="w-5 h-5 text-white" />
                                            </div>
                                            {modelLoading ? 'While AI Loads - Photo Tips' : 'Photo Tips'}
                                        </h3>
                                        <ul className="text-base text-white/80 space-y-3">
                                            <li className="flex items-start gap-3">
                                                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                                                <span>Face camera directly with neutral expression</span>
                                            </li>
                                            <li className="flex items-start gap-3">
                                                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                                                <span>Good lighting, avoid shadows</span>
                                            </li>
                                            <li className="flex items-start gap-3">
                                                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                                                <span>Remove glasses if possible</span>
                                            </li>
                                            <li className="flex items-start gap-3">
                                                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0"></span>
                                                <span>Clear, high-quality image works best</span>
                                            </li>
                                            {modelLoading && (
                                                <li className="flex items-start gap-3 mt-4 p-4 bg-blue-500/10 rounded-2xl border border-blue-500/30">
                                                    <Sparkles className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                                                    <span className="font-semibold text-blue-300">
                                                        You can upload/take photos now! Analysis will begin when the AI model is ready.
                                                    </span>
                                                </li>
                                            )}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {step === 2 && scores && (
                        <div className="space-y-8 animate-fadeInUp">
                            <div className="flex justify-between items-center mb-8">
                                <button
                                    onClick={() => { setStep(1); setImage(null); setScores(null); }}
                                    className="flex items-center gap-2 px-6 py-3 bg-white/10 backdrop-blur-sm text-white rounded-xl hover:bg-white/20 transition-all duration-300 border border-white/20 font-semibold group"
                                >
                                    <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                                    Back
                                </button>
                                <div className="text-center flex-1">
                                    {scores.isBasicMode ? (
                                        <div className="inline-flex items-center gap-3 px-6 py-3 bg-orange-500/10 backdrop-blur-sm border border-orange-500/30 rounded-2xl">
                                            <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse"></div>
                                            <div className="text-left">
                                                <p className="text-sm text-orange-300 font-semibold">
                                                    Basic Estimation Mode
                                                </p>
                                                <p className="text-xs text-orange-400/80">
                                                    AI model not loaded - using image analysis
                                                </p>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="inline-flex items-center gap-3 px-6 py-3 bg-green-500/10 backdrop-blur-sm border border-green-500/30 rounded-2xl">
                                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                                            <p className="text-sm text-green-300 font-semibold">
                                                AI Model Analysis Active
                                            </p>
                                        </div>
                                    )}
                                </div>
                                <div className="w-20"></div>
                            </div>

                            <div className="grid md:grid-cols-2 gap-8">
                                <div className="space-y-6">
                                    {image && (
                                        <div className="relative overflow-hidden rounded-3xl group">
                                            <img ref={imageRef} src={image} alt="Analyzed" className="w-full rounded-3xl shadow-2xl border border-white/20" />
                                            <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                        </div>
                                    )}

                                    <div className="relative overflow-hidden rounded-3xl bg-white/10 backdrop-blur-xl border border-white/20 shadow-2xl p-8 hover:shadow-teal-500/20 hover:border-teal-500/30 transition-all duration-500 group cursor-pointer">
                                        <div className="absolute inset-0 bg-gradient-to-br from-teal-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                        
                                        <h3 className="relative text-2xl font-bold text-white mb-6 flex items-center gap-3">
                                            <div className="w-1 h-8 bg-gradient-to-b from-teal-500 to-cyan-600 rounded-full"></div>
                                            Your Analysis
                                        </h3>
                                        
                                        <div className="relative space-y-3">
                                            {scores.isBasicMode ? (
                                                <>
                                                    <div className="p-4 bg-amber-500/10 backdrop-blur-sm border border-amber-500/30 rounded-2xl mb-4">
                                                        <p className="text-sm text-amber-300 italic">
                                                            {scores.measurements.note}
                                                        </p>
                                                    </div>
                                                    <div className="space-y-3">
                                                        <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                            <span className="text-white/70 font-medium">Brightness</span>
                                                            <span className="text-white font-bold text-lg">{scores.measurements.brightness}/255</span>
                                                        </div>
                                                        <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                            <span className="text-white/70 font-medium">Contrast</span>
                                                            <span className="text-white font-bold text-lg">{scores.measurements.contrast}</span>
                                                        </div>
                                                        <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                            <span className="text-white/70 font-medium">Texture</span>
                                                            <span className="text-white font-bold text-lg">{scores.measurements.texture}</span>
                                                        </div>
                                                        <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                            <span className="text-white/70 font-medium">Color Balance</span>
                                                            <span className="text-white font-bold text-lg">{scores.measurements.colorBalance}</span>
                                                        </div>
                                                    </div>
                                                </>
                                            ) : (
                                                <div className="space-y-3">
                                                    <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                        <span className="text-white/70 font-medium">Face Shape</span>
                                                        <span className="text-white font-bold text-lg">{scores.measurements.faceShape}</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                        <span className="text-white/70 font-medium">Eye Type</span>
                                                        <span className="text-white font-bold text-lg">{scores.measurements.eyeType}</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                        <span className="text-white/70 font-medium">Canthal Tilt</span>
                                                        <span className="text-white font-bold text-lg">{scores.measurements.canthalTilt}</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                        <span className="text-white/70 font-medium">Maxilla Development</span>
                                                        <span className="text-white font-bold text-lg">{scores.measurements.maxillaDevelopment}</span>
                                                    </div>
                                                    <div className="flex justify-between items-center bg-white/5 backdrop-blur-sm px-5 py-4 rounded-2xl border border-white/10 hover:border-white/20 transition-all">
                                                        <span className="text-white/70 font-medium">Nose Shape</span>
                                                        <span className="text-white font-bold text-lg">{scores.measurements.noseShape}</span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <div className={`relative overflow-hidden rounded-3xl bg-gradient-to-br ${currentTheme.gradient} p-1 shadow-2xl ${currentTheme.glow} animate-glow transition-all duration-1000`}>
                                        <div className={`bg-gradient-to-br ${currentTheme.bg} rounded-3xl p-8 transition-all duration-1000`}>
                                            <div className="text-center">
                                                <p className="text-white/70 text-sm font-medium mb-3 uppercase tracking-wider">Current Score</p>
                                                <div className="relative inline-block">
                                                    <p className={`text-7xl font-bold text-transparent bg-gradient-to-r ${currentTheme.text} bg-clip-text mb-4 transition-all duration-1000`}>{scores.current}</p>
                                                    <div className={`absolute inset-0 blur-2xl opacity-30 -z-10 transition-all duration-1000 ${currentTheme.orb1}`}></div>
                                                </div>
                                                
                                                <div className="w-16 h-1 bg-gradient-to-r from-transparent via-white/50 to-transparent mx-auto my-6"></div>
                                                
                                                <p className="text-white/70 text-sm font-medium mb-2 uppercase tracking-wider">Potential Score</p>
                                                <p className="text-4xl font-bold text-white mb-4">{scores.potential}</p>
                                                
                                                <p className="text-white/50 text-xs mt-6">
                                                    Based on facial symmetry & proportions
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                                        {Object.entries(scores.features).map(([key, value]) => (
                                            <ScoreCard
                                                key={key}
                                                label={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                                score={value}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <button
                                onClick={() => setStep(3)}
                                className="relative w-full group overflow-hidden"
                            >
                                <div className={`absolute inset-0 bg-gradient-to-r ${currentTheme.gradient} rounded-3xl group-hover:scale-105 transition-all duration-700`}></div>
                                <div className="relative flex items-center justify-center gap-3 py-6 text-white font-bold text-xl">
                                    <Sparkles className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                                    <span>Get AI Recommendations</span>
                                    <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
                                </div>
                                <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 rounded-3xl transition-opacity duration-500"></div>
                            </button>
                        </div>
                    )}

                    {step === 3 && scores && (
                        <div className="space-y-6 animate-fadeInUp">
                            {console.log('🎯 Rendering Step 3 - AI Tips page')}
                            {console.log('Current scores:', scores)}
                            <div className="flex justify-between items-center mb-6">
                                <button
                                    onClick={() => {
                                        console.log('Back button clicked');
                                        setStep(2);
                                    }}
                                    className="flex items-center gap-2 px-6 py-3 bg-white/10 backdrop-blur-sm text-white rounded-xl hover:bg-white/20 transition-all duration-300 border border-white/20 font-semibold group"
                                >
                                    <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                                    Back
                                </button>
                                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                                    <Sparkles className="w-6 h-6 text-teal-400" />
                                    Personalized Improvement Plan
                                </h2>
                                <button
                                    onClick={async (event) => {
                                        console.log('🚨 BUTTON CLICK DETECTED!');
                                        console.log('Event object:', event);
                                        console.log('Current step:', step);
                                        console.log('loadingRecommendations state:', loadingRecommendations);
                                        
                                        console.log('🔥 AI Tips button clicked!');
                                        console.log('Scores object:', scores);
                                        console.log('Scores features:', scores?.features);
                                        
                                        if (!scores || !scores.features) {
                                            console.error('❌ No scores available!');
                                            alert('No scores available! Please go back and analyze a photo first.');
                                            return;
                                        }
                                        
                                        setLoadingRecommendations(true);
                                        console.log('✅ Set loading to true');
                                        const newRecommendations = {};
                                        
                                        // Generate AI recommendations for each low-scoring feature
                                        const lowScoringFeatures = Object.entries(scores.features).filter(([_, score]) => score < 80);
                                        console.log('Low scoring features:', lowScoringFeatures);
                                        
                                        if (lowScoringFeatures.length === 0) {
                                            console.log('🎉 No low scoring features - all scores are above 80!');
                                            alert('Congratulations! All your features score above 80. No improvements needed!');
                                            setLoadingRecommendations(false);
                                            return;
                                        }
                                        
                                        for (const [feature, score] of lowScoringFeatures) {
                                            console.log(`🤖 Getting AI recommendations for ${feature} (score: ${score})`);
                                            try {
                                                newRecommendations[feature] = await getAIRecommendations(
                                                    feature, 
                                                    score, 
                                                    scores.measurements
                                                );
                                                console.log(`✅ Got recommendations for ${feature}:`, newRecommendations[feature]);
                                            } catch (error) {
                                                console.log(`🔄 Using enhanced recommendations for ${feature}`);
                                                newRecommendations[feature] = getFallbackRecommendations(feature, score);
                                                console.log(`✨ Generated recommendations for ${feature}:`, newRecommendations[feature]);
                                            }
                                        }
                                        
                                        console.log('Final recommendations:', newRecommendations);
                                        setAiRecommendations(newRecommendations);
                                        setLoadingRecommendations(false);
                                        console.log('✅ Process completed');
                                    }}
                                    disabled={loadingRecommendations}
                                    className="bg-gradient-to-r from-rose-500 to-pink-500 text-white px-8 py-3 rounded-xl hover:from-rose-600 hover:to-pink-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center font-semibold shadow-md hover:shadow-lg"
                                    style={{ pointerEvents: 'auto' }}
                                >
                                    {loadingRecommendations ? (
                                        <>
                                            <Loader className="w-4 h-4 mr-2 animate-spin" />
                                            Generating AI Tips...
                                        </>
                                    ) : (
                                        <>
                                            <Sparkles className="w-4 h-4 mr-2" />
                                            Get AI Tips
                                        </>
                                    )}
                                </button>
                            </div>

                            <div className="space-y-6">
                                {(() => {
                                    const lowScoringFeatures = Object.entries(scores.features).filter(([_, score]) => score < 80);
                                    console.log('🔍 Low scoring features for recommendations:', lowScoringFeatures);
                                    return lowScoringFeatures;
                                })()
                                    .sort(([_, a], [__, b]) => a - b)
                                    .map(([feature, score]) => {
                                        const recommendations = aiRecommendations[feature] || getFallbackRecommendations(feature, score);
                                        return (
                                            <div key={feature} className="bg-white p-6 rounded-2xl border border-gray-200 hover:shadow-lg transition-all duration-300">
                                                <div className="flex items-start justify-between mb-4">
                                                    <div className="flex items-center">
                                                        <div className={`w-3 h-3 rounded-full mr-3 ${
                                                            score >= 80 ? 'bg-emerald-500' :
                                                            score >= 60 ? 'bg-amber-500' :
                                                            'bg-rose-500'
                                                        }`}></div>
                                                        <h3 className="text-xl font-semibold text-gray-900">
                                                            {feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                                        </h3>
                                                    </div>
                                                    <div className="text-right">
                                                        <span className="text-2xl font-bold text-gray-900">
                                                            {score}
                                                        </span>
                                                        <span className="text-gray-500 text-sm block">
                                                            / 100
                                                        </span>
                                                    </div>
                                                </div>

                                                <p className="text-gray-600 text-sm mb-4 leading-relaxed">
                                                    {feature === 'jawline' ? 'Chewing and performing facial exercises will give you a much stronger jawline.' :
                                                     feature === 'skinQuality' ? 'Skincare routines are crucial for everybody. Nobody is above it, you should start today.' :
                                                     feature === 'eyeArea' ? 'Eye care and proper sleep can dramatically improve your appearance.' :
                                                     feature === 'cheekbones' ? 'Facial exercises and body fat reduction enhance cheekbone definition.' :
                                                     'Targeted improvements can enhance this facial feature significantly.'}
                                                </p>

                                                <div className="space-y-3">
                                                    {recommendations.slice(0, 4).map((rec, idx) => (
                                                        <div 
                                                            key={idx} 
                                                            onClick={() => setExpandedRecommendation({ feature, recommendationIndex: idx, recommendation: rec })}
                                                            className="flex items-start bg-gray-50 p-4 rounded-xl border border-gray-100 hover:bg-rose-50 hover:border-rose-300 cursor-pointer transition-all duration-300 group"
                                                        >
                                                            <span className="text-white bg-rose-500 rounded-full w-7 h-7 flex items-center justify-center text-xs font-semibold mr-3 mt-0.5 flex-shrink-0 group-hover:scale-110 transition-transform">
                                                                {idx + 1}
                                                            </span>
                                                            <div className="flex-1">
                                                                <span className="text-gray-700 text-sm leading-relaxed">{rec}</span>
                                                                <div className="text-xs text-rose-500 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                    Click for detailed guide →
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                    {aiRecommendations[feature] && (
                                                        <div className="flex items-center justify-between pt-2">
                                                            <span className="text-xs text-emerald-600 flex items-center font-medium">
                                                                <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2"></span>
                                                                AI-Generated & Personalized
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>

                                                    {feature === 'jawline' && (
                                                        <div className="bg-white p-4 rounded-lg">
                                                            <h4 className="font-semibold text-gray-700 mb-3">💪 Jawline Exercises:</h4>
                                                            <div className="grid grid-cols-3 gap-3">
                                                                <div className="text-center">
                                                                    <div className="bg-indigo-100 rounded-lg p-4 mb-2">
                                                                        <div className="text-4xl">🧘</div>
                                                                    </div>
                                                                    <p className="text-xs text-gray-600 font-medium">Chin Lifts</p>
                                                                    <p className="text-xs text-gray-500">3×15 daily</p>
                                                                </div>
                                                                <div className="text-center">
                                                                    <div className="bg-indigo-100 rounded-lg p-4 mb-2">
                                                                        <div className="text-4xl">😬</div>
                                                                    </div>
                                                                    <p className="text-xs text-gray-600 font-medium">Jaw Clench</p>
                                                                    <p className="text-xs text-gray-500">Hold 10s</p>
                                                                </div>
                                                                <div className="text-center">
                                                                    <div className="bg-indigo-100 rounded-lg p-4 mb-2">
                                                                        <div className="text-4xl">💪</div>
                                                                    </div>
                                                                    <p className="text-xs text-gray-600 font-medium">Neck Curls</p>
                                                                    <p className="text-xs text-gray-500">3×15 reps</p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {feature === 'skinQuality' && (
                                                        <div className="bg-white p-4 rounded-lg">
                                                            <h4 className="font-semibold text-gray-700 mb-3">🧴 Optimized Skincare:</h4>
                                                            <div className="space-y-2">
                                                                <div className="flex items-center">
                                                                    <span className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded text-sm font-medium mr-3">AM</span>
                                                                    <span className="text-sm text-gray-700">Cleanser → Vitamin C → Moisturizer → SPF 50</span>
                                                                </div>
                                                                <div className="flex items-center">
                                                                    <span className="bg-purple-100 text-purple-700 px-3 py-1 rounded text-sm font-medium mr-3">PM</span>
                                                                    <span className="text-sm text-gray-700">Cleanser → Retinol 0.5% → Moisturizer</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {feature === 'cheekbones' && (
                                                        <div className="bg-white p-4 rounded-lg">
                                                            <h4 className="font-semibold text-gray-700 mb-3">🎯 Cheekbone Enhancement:</h4>
                                                            <div className="grid grid-cols-2 gap-3">
                                                                <div className="bg-purple-50 p-3 rounded-lg">
                                                                    <p className="font-medium text-sm mb-1">Cheek Puffs</p>
                                                                    <p className="text-xs text-gray-600">Hold air, switch sides</p>
                                                                    <p className="text-xs text-indigo-600 mt-1">3 sets × 10 reps</p>
                                                                </div>
                                                                <div className="bg-purple-50 p-3 rounded-lg">
                                                                    <p className="font-medium text-sm mb-1">Fish Face</p>
                                                                    <p className="text-xs text-gray-600">Suck cheeks, hold 5s</p>
                                                                    <p className="text-xs text-indigo-600 mt-1">3 sets × 15 reps</p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                
                                                {feature === 'jawline' && (
                                                    <div className="bg-gray-700/30 p-4 rounded-xl mt-4">
                                                        <h4 className="font-bold text-white mb-3">💪 Jawline Exercises:</h4>
                                                        <div className="grid grid-cols-3 gap-3">
                                                            <div className="text-center">
                                                                <div className="bg-gray-600/50 rounded-lg p-4 mb-2">
                                                                    <div className="text-3xl">😤</div>
                                                                </div>
                                                                <p className="text-xs text-gray-300 font-medium">Chin Lifts</p>
                                                                <p className="text-xs text-gray-400">3×15 daily</p>
                                                            </div>
                                                            <div className="text-center">
                                                                <div className="bg-gray-600/50 rounded-lg p-4 mb-2">
                                                                    <div className="text-3xl">😬</div>
                                                                </div>
                                                                <p className="text-xs text-gray-300 font-medium">Jaw Clench</p>
                                                                <p className="text-xs text-gray-400">Hold 10s</p>
                                                            </div>
                                                            <div className="text-center">
                                                                <div className="bg-gray-600/50 rounded-lg p-4 mb-2">
                                                                    <div className="text-3xl">💪</div>
                                                                </div>
                                                                <p className="text-xs text-gray-300 font-medium">Neck Curls</p>
                                                                <p className="text-xs text-gray-400">3×15 reps</p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                            </div>

                            <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-200">
                                <h3 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
                                    <span className="text-3xl mr-3">🎯</span> Personalized Action Plan
                                </h3>
                                {(() => {
                                    const actionPlan = generateActionPlan(aiRecommendations, scores);
                                    const hasRecommendations = aiRecommendations && Object.keys(aiRecommendations).length > 0;
                                    
                                    return (
                                        <div className="space-y-4">
                                            {hasRecommendations && (
                                                <div className="bg-blue-50 p-4 rounded-xl border border-blue-200">
                                                    <p className="text-sm text-blue-800 font-medium">
                                                        📊 Based on your {Object.keys(aiRecommendations).length} improvement areas and {Object.values(aiRecommendations).flat().length} personalized recommendations
                                                    </p>
                                                </div>
                                            )}
                                            <div className="space-y-3">
                                                <div className="flex items-start bg-gray-50 p-5 rounded-xl border border-gray-200">
                                                    <span className="bg-rose-500 text-white w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold mr-4 flex-shrink-0">1</span>
                                                    <div className="text-gray-700">
                                                        <span className="text-rose-600 font-semibold">Week 1:</span>
                                                        <span className="ml-2">{actionPlan.week1}</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-start bg-gray-50 p-5 rounded-xl border border-gray-200">
                                                    <span className="bg-rose-500 text-white w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold mr-4 flex-shrink-0">2</span>
                                                    <div className="text-gray-700">
                                                        <span className="text-rose-600 font-semibold">Week 2:</span>
                                                        <span className="ml-2">{actionPlan.week2}</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-start bg-gray-50 p-5 rounded-xl border border-gray-200">
                                                    <span className="bg-rose-500 text-white w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold mr-4 flex-shrink-0">3</span>
                                                    <div className="text-gray-700">
                                                        <span className="text-rose-600 font-semibold">Month 1:</span>
                                                        <span className="ml-2">{actionPlan.month1}</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-start bg-gray-50 p-5 rounded-xl border border-gray-200">
                                                    <span className="bg-rose-500 text-white w-9 h-9 rounded-full flex items-center justify-center text-sm font-semibold mr-4 flex-shrink-0">4</span>
                                                    <div className="text-gray-700">
                                                        <span className="text-rose-600 font-semibold">Month 2+:</span>
                                                        <span className="ml-2">{actionPlan.month2}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            {hasRecommendations && (
                                                <div className="bg-amber-50 p-4 rounded-xl border border-amber-200 mt-4">
                                                    <p className="text-sm text-amber-900">
                                                        💡 <strong className="font-semibold">Tip:</strong> Focus on 2-3 recommendations per week for best results. Consistency beats intensity!
                                                    </p>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })()}
                            </div>

                            <button
                                onClick={() => {
                                    setStep(1);
                                    setImage(null);
                                    setScores(null);
                                }}
                                className="relative w-full group overflow-hidden"
                            >
                                <div className={`absolute inset-0 bg-gradient-to-r ${currentTheme.gradient} rounded-3xl group-hover:scale-105 transition-all duration-700`}></div>
                                <div className="relative flex items-center justify-center gap-3 py-6 text-white font-bold text-xl">
                                    <Sparkles className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                                    <span>Start New Analysis</span>
                                    <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
                                </div>
                                <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 rounded-3xl transition-opacity duration-500"></div>
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Detailed Recommendation Modal */}
            {expandedRecommendation && (
                <div 
                    className="fixed inset-0 bg-black/80 backdrop-blur-xl flex items-center justify-center z-50 p-4 animate-fadeInUp"
                    onClick={() => setExpandedRecommendation(null)}
                >
                    <div 
                        className={`bg-gradient-to-br ${currentTheme.bg} rounded-3xl max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-white/20`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Modal Header */}
                        <div className="sticky top-0 bg-white/10 backdrop-blur-xl border-b border-white/20 p-6 flex justify-between items-center z-10">
                            <div>
                                <h2 className="text-3xl font-bold text-white mb-1">
                                    {detailedGuide?.title || `Guide: ${expandedRecommendation.feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}`}
                                </h2>
                                <p className="text-white/60 text-sm">Expert recommendations for improvement</p>
                            </div>
                            <button 
                                onClick={() => setExpandedRecommendation(null)}
                                className="w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 transition-all flex items-center justify-center"
                            >
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        <div className="p-8 space-y-8">
                            {/* Recommendation Overview */}
                            <div className={`bg-gradient-to-r ${currentTheme.primary} p-6 rounded-3xl`}>
                                <p className="text-white font-semibold text-lg">{expandedRecommendation.recommendation}</p>
                            </div>

                            {loadingDetailedGuide ? (
                                <div className="flex flex-col items-center justify-center py-12">
                                    <Loader className="w-12 h-12 text-white animate-spin mb-4" />
                                    <p className="text-white/70">Generating detailed guide with AI...</p>
                                </div>
                            ) : detailedGuide ? (
                                <>
                                    {/* Overview */}
                                    {detailedGuide.overview && (
                                        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-6">
                                            <p className="text-white/90 leading-relaxed">{detailedGuide.overview}</p>
                                        </div>
                                    )}

                                    {/* Step-by-Step Guide */}
                                    {detailedGuide.steps && detailedGuide.steps.length > 0 && (
                                        <div className="space-y-4">
                                            <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                <span className={`w-1 h-8 bg-gradient-to-b ${currentTheme.primary} rounded-full`}></span>
                                                Step-by-Step Guide
                                            </h3>
                                            <div className="space-y-4">
                                                {detailedGuide.steps.map((step, index) => (
                                                    <div key={index} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all duration-300">
                                                        <div className="flex items-start gap-4">
                                                            <div className={`flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br ${currentTheme.primary} flex items-center justify-center`}>
                                                                <span className="text-white font-bold text-xl">{step.stepNumber}</span>
                                                            </div>
                                                            <div className="flex-1">
                                                                <h4 className="text-xl font-bold text-white mb-2">{step.title}</h4>
                                                                <p className="text-white/80 mb-4">{step.description}</p>
                                                                <div className="flex flex-wrap gap-3 mb-3">
                                                                    <span className="inline-flex items-center gap-2 bg-white/5 px-3 py-1 rounded-full text-sm text-white/70">
                                                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                                        </svg>
                                                                        {step.duration}
                                                                    </span>
                                                                    <span className="inline-flex items-center gap-2 bg-white/5 px-3 py-1 rounded-full text-sm text-white/70">
                                                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                                                        </svg>
                                                                        {step.frequency}
                                                                    </span>
                                                                </div>
                                                                {step.tips && step.tips.length > 0 && (
                                                                    <div className="mt-3 space-y-1">
                                                                        {step.tips.map((tip, i) => (
                                                                            <p key={i} className="text-sm text-white/60 flex items-start gap-2">
                                                                                <span className={`text-${currentTheme.accent}`}>💡</span>
                                                                                {tip}
                                                                            </p>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Expected Results Timeline */}
                                    {detailedGuide.expectedResults && (
                                        <div className="space-y-4">
                                            <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                <span className={`w-1 h-8 bg-gradient-to-b ${currentTheme.primary} rounded-full`}></span>
                                                Expected Results
                                            </h3>
                                            <div className="grid md:grid-cols-3 gap-4">
                                                {detailedGuide.expectedResults.week1 && (
                                                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                                                        <div className={`text-2xl font-bold text-transparent bg-gradient-to-r ${currentTheme.text} bg-clip-text mb-2`}>Week 1</div>
                                                        <p className="text-white/70 text-sm">{detailedGuide.expectedResults.week1}</p>
                                                    </div>
                                                )}
                                                {detailedGuide.expectedResults.week4 && (
                                                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                                                        <div className={`text-2xl font-bold text-transparent bg-gradient-to-r ${currentTheme.text} bg-clip-text mb-2`}>Week 4</div>
                                                        <p className="text-white/70 text-sm">{detailedGuide.expectedResults.week4}</p>
                                                    </div>
                                                )}
                                                {detailedGuide.expectedResults.week8 && (
                                                    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                                                        <div className={`text-2xl font-bold text-transparent bg-gradient-to-r ${currentTheme.text} bg-clip-text mb-2`}>Week 8+</div>
                                                        <p className="text-white/70 text-sm">{detailedGuide.expectedResults.week8}</p>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Do's and Don'ts */}
                                    {(detailedGuide.dos || detailedGuide.donts) && (
                                        <div className="grid md:grid-cols-2 gap-6">
                                            {detailedGuide.dos && detailedGuide.dos.length > 0 && (
                                                <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                                                    <h4 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                                        <span className="text-2xl">✅</span>
                                                        Do This
                                                    </h4>
                                                    <ul className="space-y-2">
                                                        {detailedGuide.dos.map((item, i) => (
                                                            <li key={i} className="text-white/80 text-sm flex items-start gap-2">
                                                                <span className="text-green-400 mt-1">●</span>
                                                                {item}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                            {detailedGuide.donts && detailedGuide.donts.length > 0 && (
                                                <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
                                                    <h4 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                                        <span className="text-2xl">❌</span>
                                                        Avoid This
                                                    </h4>
                                                    <ul className="space-y-2">
                                                        {detailedGuide.donts.map((item, i) => (
                                                            <li key={i} className="text-white/80 text-sm flex items-start gap-2">
                                                                <span className="text-red-400 mt-1">●</span>
                                                                {item}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Pro Tips */}
                                    {detailedGuide.proTips && detailedGuide.proTips.length > 0 && (
                                        <div className={`bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-2xl p-6`}>
                                            <h4 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                                <span className="text-2xl">💡</span>
                                                Pro Tips
                                            </h4>
                                            <div className="space-y-2">
                                                {detailedGuide.proTips.map((tip, i) => (
                                                    <p key={i} className="text-white/90 text-sm flex items-start gap-2">
                                                        <span className="text-amber-400">▸</span>
                                                        {tip}
                                                    </p>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Related Products */}
                                    {detailedGuide.relatedProducts && detailedGuide.relatedProducts.length > 0 && (
                                        <div className="space-y-4">
                                            <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                                                <span className={`w-1 h-8 bg-gradient-to-b ${currentTheme.primary} rounded-full`}></span>
                                                Recommended Products
                                            </h3>
                                            <div className="grid md:grid-cols-3 gap-4">
                                                {detailedGuide.relatedProducts.map((product, i) => (
                                                    <div key={i} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-4 hover:bg-white/10 transition-all">
                                                        <p className="text-white/80 text-sm">{product}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </>
                            ) : null}
                        </div>

                        <div className="sticky bottom-0 bg-white/5 backdrop-blur-sm border-t border-white/10 p-6">
                            <button
                                onClick={() => setExpandedRecommendation(null)}
                                className={`w-full bg-gradient-to-r ${currentTheme.gradient} text-white py-4 rounded-2xl font-bold text-lg hover:scale-105 transition-all duration-300 shadow-xl`}
                            >
                                Got it! Back to Recommendations ✓
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}



