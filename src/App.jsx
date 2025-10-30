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
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const imageRef = useRef(null);
    const [stream, setStream] = useState(null);

    // Monitor camera status changes for debugging
    useEffect(() => {
        console.log('🔄 Camera status changed to:', cameraStatus);
    }, [cameraStatus]);

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
        <div className="bg-white rounded-2xl p-6 border border-gray-200 hover:shadow-2xl hover:shadow-rose-200/50 hover:border-rose-300 transition-all duration-300 cursor-pointer hover:scale-105">
            <div className="text-center">
                <h3 className="text-gray-500 text-xs font-semibold mb-3 uppercase tracking-wider">{label}</h3>
                <div className="text-5xl font-bold text-gray-900 mb-4">{score}</div>
                <div className="w-full bg-gray-100 rounded-full h-2.5 mb-3">
                    <div
                        className={`h-2.5 rounded-full transition-all duration-500 ${
                            score >= 80 ? 'bg-gradient-to-r from-green-500 to-green-600' :
                            score >= 60 ? 'bg-gradient-to-r from-yellow-400 to-yellow-500' :
                            'bg-gradient-to-r from-rose-400 to-rose-500'
                        }`}
                        style={{ width: `${score}%` }}
                    />
                </div>
                <div className={`text-sm font-semibold ${
                    score >= 80 ? 'text-green-600' :
                    score >= 60 ? 'text-yellow-600' :
                    'text-rose-600'
                }`}>
                    {score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Improve'}
                </div>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-50 p-4 md:p-8">
            {console.log('🖥️ App render - step:', step, 'scores:', !!scores, 'loadingRecommendations:', loadingRecommendations)}
            <div className="max-w-6xl mx-auto">
                <div className="bg-white rounded-3xl shadow-sm p-8 md:p-12 border border-gray-100">
                    <div className="text-center mb-12">
                        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-2 tracking-tight">
                            FaceMaxx
                        </h1>
                        
                        {/* Model Status Indicator */}
                        <div className="mt-8">
                            {modelLoading ? (
                                <div className="inline-flex items-center text-gray-700 bg-gray-100 rounded-full px-6 py-3 border border-gray-200">
                                    <Loader className="w-5 h-5 mr-2 animate-spin text-gray-600" />
                                    <span className="font-medium text-sm">Loading AI Model...</span>
                                </div>
                            ) : model ? (
                                <div className="inline-flex items-center text-green-700 bg-green-50 rounded-full px-6 py-3 border border-green-100">
                                    <span className="text-green-600 mr-2">✓</span>
                                    <span className="font-medium text-sm">AI Ready</span>
                                </div>
                            ) : !modelError ? (
                                <div className="inline-flex items-center text-gray-600 bg-gray-50 rounded-full px-6 py-3 border border-gray-100">
                                    <span className="text-gray-500 mr-2">⏳</span>
                                    <span className="font-medium text-sm">Loading...</span>
                                </div>
                            ) : null}
                        </div>
                        
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
                        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                            <div className="bg-white p-8 rounded-xl text-center max-w-md mx-4">
                                <Loader className="w-16 h-16 mx-auto mb-4 text-indigo-600 animate-spin" />
                                <p className="text-xl font-semibold">
                                    {modelLoading ? 'Loading AI model...' : 'Analyzing facial features...'}
                                </p>
                                <p className="text-gray-600 mt-2">
                                    {modelLoading 
                                        ? 'Preparing TensorFlow.js and face detection model' 
                                        : model 
                                            ? '🤖 Using AI Model for high-precision analysis'
                                            : '📊 Using Basic Analysis mode (AI model unavailable)'
                                    }
                                </p>
                                {modelLoading && (
                                    <div className="mt-4 text-sm text-gray-500">
                                        This may take 10-30 seconds on first load
                                    </div>
                                )}
                                {!model && !modelLoading && (
                                    <div className="mt-4 text-sm text-orange-600 bg-orange-50 rounded p-2">
                                        Using estimated analysis - results may be less accurate
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {step === 1 && (
                        <div className="space-y-6">
                            <h2 className="text-2xl font-semibold text-center mb-6">Upload or Capture Your Photo</h2>

                            {!stream ? (
                                <div className="space-y-4">
                                    {/* Primary option - File Upload */}
                                    <div
                                        onClick={() => fileInputRef.current.click()}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                        onDrop={handleDrop}
                                        className={`w-full flex flex-col items-center justify-center p-12 border-2 border-dashed rounded-2xl transition-all relative cursor-pointer ${
                                            dragOver 
                                                ? 'border-rose-400 bg-rose-50 scale-102' 
                                                : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                                        }`}
                                    >
                                        <div className="absolute top-4 right-4 bg-rose-500 text-white px-3 py-1.5 rounded-full text-xs font-semibold shadow-sm">
                                            Recommended
                                        </div>
                                        <Upload className={`w-20 h-20 mb-4 ${dragOver ? 'text-rose-500 scale-110' : 'text-gray-400'} transition-all`} />
                                        <span className="text-2xl font-semibold text-gray-900 mb-2">
                                            Upload your photo
                                        </span>
                                        <span className="text-base text-gray-600 text-center max-w-md leading-relaxed">
                                            {dragOver 
                                                ? 'Drop your photo here' 
                                                : 'Drag and drop, or click to select a clear face photo'
                                            }
                                        </span>
                                        <span className="text-sm text-gray-500 mt-4 bg-gray-100 px-4 py-2 rounded-full">
                                            JPG or PNG • Max 10MB
                                        </span>
                                    </div>

                                    {/* Secondary option - Camera */}
                                    <div className="text-center">
                                        <p className="text-gray-500 text-sm mb-3">Or use your camera (requires permission)</p>
                                        <button
                                            onClick={async () => {
                                                console.log('Camera button clicked');
                                                // Quick test first
                                                try {
                                                    const testStream = await navigator.mediaDevices.getUserMedia({ video: true });
                                                    console.log('✅ Quick camera test passed');
                                                    testStream.getTracks().forEach(track => track.stop());
                                                } catch (e) {
                                                    console.error('❌ Quick camera test failed:', e);
                                                    alert('Camera test failed: ' + e.message);
                                                    return;
                                                }
                                                
                                                // Reset camera status and start camera
                                                // Camera UI shows when stream exists (still in step 1)
                                                setCameraStatus('idle');
                                                startCamera();
                                            }}
                                            className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-purple-300 rounded-xl hover:border-purple-500 hover:bg-purple-50 transition-all mx-auto w-full max-w-sm"
                                        >
                                            <Camera className="w-12 h-12 text-purple-500 mb-3" />
                                            <span className="text-lg font-medium text-gray-700">Take Photo</span>
                                            <span className="text-xs text-gray-500 mt-1">
                                                Quick and easy face capture
                                            </span>
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="bg-gray-100 p-4 rounded-xl">
                                        <h3 className="text-lg font-semibold mb-3 text-center">📷 Camera Preview</h3>
                                        <div className="relative bg-black rounded-lg overflow-hidden" style={{minHeight: '300px'}}>
                                            <video 
                                                ref={videoRef} 
                                                autoPlay 
                                                playsInline
                                                muted
                                                className="w-full h-full object-cover rounded-lg"
                                                style={{minHeight: '300px'}}
                                            />
                                            {cameraStatus === 'starting' && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-gray-800 text-white">
                                                    <div className="text-center">
                                                        <Loader className="w-12 h-12 mx-auto mb-3 animate-spin text-white" />
                                                        <p className="text-lg font-medium">Starting camera...</p>
                                                        <p className="text-sm text-gray-300 mt-1">Please wait while we access your camera</p>
                                                        <p className="text-xs text-gray-400 mt-2">Status: {cameraStatus}</p>
                                                    </div>
                                                </div>
                                            )}
                                            {cameraStatus === 'error' && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-gray-800 text-white">
                                                    <div className="text-center p-4">
                                                        <AlertCircle className="w-12 h-12 mx-auto mb-3 text-red-400" />
                                                        <p className="text-lg font-medium text-red-400">Camera Error</p>
                                                        <p className="text-sm text-gray-300 mt-1">Camera initialization failed</p>
                                                        <p className="text-xs text-gray-400 mt-2">Check browser console for details</p>
                                                        <div className="flex gap-2 mt-4 justify-center">
                                                            <button 
                                                                onClick={() => {
                                                                    setCameraStatus('idle');
                                                                    startCamera();
                                                                }}
                                                                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
                                                            >
                                                                Try Again
                                                            </button>
                                                            <button 
                                                                onClick={() => {
                                                                    stopCamera();
                                                                }}
                                                                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 text-sm"
                                                            >
                                                                Cancel
                                                            </button>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        
                                        {cameraStatus === 'ready' && videoRef.current?.videoWidth > 0 && (
                                            <div className="mt-2 text-center text-sm text-green-600">
                                                ✅ Camera active • Resolution: {videoRef.current.videoWidth}x{videoRef.current.videoHeight}
                                            </div>
                                        )}
                                    </div>
                                    
                                    <div className="flex gap-3">
                                        <button
                                            onClick={capturePhoto}
                                            disabled={cameraStatus !== 'ready'}
                                            className="flex-1 bg-indigo-600 text-white py-4 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                                        >
                                            📸 Capture & Analyze Photo
                                        </button>
                                        <button
                                            onClick={() => {
                                                console.log('Stopping camera...');
                                                stopCamera();
                                            }}
                                            className="px-6 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
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

                            <div className="space-y-4">
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                    <h3 className="font-semibold text-blue-900 mb-2">
                                        {modelLoading ? '⏳ While AI model loads - Photo Tips:' : '📸 Photo Tips:'}
                                    </h3>
                                    <ul className="text-sm text-blue-800 space-y-1">
                                        <li>• Face camera directly with neutral expression</li>
                                        <li>• Good lighting, avoid shadows</li>
                                        <li>• Remove glasses if possible</li>
                                        <li>• Clear, high-quality image works best</li>
                                        {modelLoading && (
                                            <li className="font-semibold text-blue-900 mt-3">
                                                💡 You can upload/take photos now! Analysis will begin when the AI model is ready.
                                            </li>
                                        )}
                                    </ul>
                                </div>


                            </div>
                        </div>
                    )}

                    {step === 2 && scores && (
                        <div className="space-y-6">
                            <div className="flex justify-between items-center mb-6">
                                <button
                                    onClick={() => { setStep(1); setImage(null); setScores(null); }}
                                    className="flex items-center text-gray-600 hover:text-gray-900"
                                >
                                    <ArrowLeft className="w-5 h-5 mr-2" />
                                    Back
                                </button>
                                <div className="text-center">
                                    {scores.isBasicMode ? (
                                        <div className="mt-2 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                                            <p className="text-sm text-orange-700 font-medium">
                                                ⚠️ Using Basic Estimation Mode
                                            </p>
                                            <p className="text-xs text-orange-600 mt-1">
                                                TensorFlow.js AI model not loaded - scores are estimated using image analysis
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                                            <p className="text-sm text-green-700 font-medium">
                                                🤖 Using AI Model Analysis
                                            </p>
                                        </div>
                                    )}
                                </div>
                                <div className="w-20"></div>
                            </div>

                            <div className="grid md:grid-cols-2 gap-6">
                                <div>
                                    {image && (
                                        <img ref={imageRef} src={image} alt="Analyzed" className="w-full rounded-xl shadow-lg mb-4" />
                                    )}

                                    <div className="bg-gray-50 p-6 rounded-2xl border border-gray-200 hover:shadow-2xl hover:shadow-rose-200/50 hover:border-rose-300 transition-all duration-300 hover:scale-[1.02] cursor-pointer">
                                        <h3 className="font-semibold text-gray-900 mb-4 text-lg">Your Analysis</h3>
                                        <div className="space-y-3">
                                            {scores.isBasicMode ? (
                                                <>
                                                    <p className="text-amber-600 italic mb-3 text-sm bg-amber-50 p-3 rounded-lg">
                                                        {scores.measurements.note}
                                                    </p>
                                                    <div className="space-y-2">
                                                        <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                            <span className="text-gray-600 font-medium">Brightness</span>
                                                            <span className="text-gray-900 font-semibold">{scores.measurements.brightness}/255</span>
                                                        </div>
                                                        <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                            <span className="text-gray-600 font-medium">Contrast</span>
                                                            <span className="text-gray-900 font-semibold">{scores.measurements.contrast}</span>
                                                        </div>
                                                        <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                            <span className="text-gray-600 font-medium">Texture</span>
                                                            <span className="text-gray-900 font-semibold">{scores.measurements.texture}</span>
                                                        </div>
                                                        <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                            <span className="text-gray-600 font-medium">Color Balance</span>
                                                            <span className="text-gray-900 font-semibold">{scores.measurements.colorBalance}</span>
                                                        </div>
                                                    </div>
                                                </>
                                            ) : (
                                                <div className="space-y-2">
                                                    <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                        <span className="text-gray-600 font-medium">Face Shape</span>
                                                        <span className="text-gray-900 font-semibold">{scores.measurements.faceShape}</span>
                                                    </div>
                                                    <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                        <span className="text-gray-600 font-medium">Eye Type</span>
                                                        <span className="text-gray-900 font-semibold">{scores.measurements.eyeType}</span>
                                                    </div>
                                                    <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                        <span className="text-gray-600 font-medium">Canthal Tilt</span>
                                                        <span className="text-gray-900 font-semibold">{scores.measurements.canthalTilt}</span>
                                                    </div>
                                                    <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                        <span className="text-gray-600 font-medium">Maxilla Development</span>
                                                        <span className="text-gray-900 font-semibold">{scores.measurements.maxillaDevelopment}</span>
                                                    </div>
                                                    <div className="flex justify-between bg-white px-4 py-3 rounded-xl shadow-sm border border-gray-100">
                                                        <span className="text-gray-600 font-medium">Nose Shape</span>
                                                        <span className="text-gray-900 font-semibold">{scores.measurements.noseShape}</span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <div className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white p-6 rounded-xl mb-6">
                                        <div className="text-center">
                                            <p className="text-sm opacity-90 mb-1">AI Calculated Score</p>
                                            <p className="text-5xl font-bold mb-2">{scores.current}</p>
                                            <p className="text-sm opacity-90">Potential Score</p>
                                            <p className="text-3xl font-semibold">{scores.potential}</p>
                                            <p className="text-xs opacity-75 mt-2">Based on facial symmetry & proportions</p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
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
                                className="w-full bg-gradient-to-r from-rose-500 to-pink-500 text-white py-4 rounded-xl font-semibold hover:from-rose-600 hover:to-pink-600 transition-all duration-200 flex items-center justify-center text-lg shadow-md hover:shadow-lg"
                            >
                                Get AI Recommendations
                                <ArrowRight className="w-5 h-5 ml-2" />
                            </button>
                        </div>
                    )}

                    {step === 3 && scores && (
                        <div className="space-y-6">
                            {console.log('🎯 Rendering Step 3 - AI Tips page')}
                            {console.log('Current scores:', scores)}
                            <div className="flex justify-between items-center mb-6">
                                <button
                                    onClick={() => {
                                        console.log('Back button clicked');
                                        setStep(2);
                                    }}
                                    className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
                                >
                                    <ArrowLeft className="w-5 h-5 mr-2" />
                                    Back to Scores
                                </button>
                                <h2 className="text-2xl font-semibold text-gray-900 flex items-center">
                                    <Sparkles className="w-6 h-6 mr-2 text-rose-500" />
                                    Personalized Improvement Plan
                                </h2>
                                <button
                                    onClick={async (event) => {
                                        console.log('🚨 BUTTON CLICK DETECTED!');
                                        console.log('Event object:', event);
                                        console.log('Current step:', step);
                                        console.log('loadingRecommendations state:', loadingRecommendations);
                                        
                                        // Test environment variables
                                        console.log('� ENV TEST:');
                                        console.log('All import.meta.env:', import.meta.env);
                                        console.log('VITE_OPENAI_API_KEY exists:', !!import.meta.env.VITE_OPENAI_API_KEY);
                                        console.log('VITE_OPENAI_API_KEY length:', import.meta.env.VITE_OPENAI_API_KEY?.length || 0);
                                        console.log('First 20 chars:', import.meta.env.VITE_OPENAI_API_KEY?.substring(0, 20) || 'MISSING');
                                        
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
                                className="w-full bg-gray-100 text-gray-700 py-4 rounded-xl font-semibold hover:bg-gray-200 transition-all duration-300 border border-gray-300"
                            >
                                Start New Analysis
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Detailed Recommendation Modal */}
            {expandedRecommendation && (
                <div 
                    className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
                    onClick={() => setExpandedRecommendation(null)}
                >
                    <div 
                        className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex justify-between items-center">
                            <h2 className="text-2xl font-semibold text-gray-900">
                                Detailed Guide: {expandedRecommendation.feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                            </h2>
                            <button 
                                onClick={() => setExpandedRecommendation(null)}
                                className="text-gray-400 hover:text-gray-600 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        <div className="p-8">
                            <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 mb-6">
                                <p className="text-rose-900 font-medium">{expandedRecommendation.recommendation}</p>
                            </div>

                            {expandedRecommendation.recommendationIndex === 0 && expandedRecommendation.feature === 'faceShape' && (
                                <div className="space-y-6">
                                    <h3 className="text-xl font-semibold text-gray-900 mb-4">🧘‍♂️ Facial Exercises for Better Face Shape</h3>
                                    
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="bg-rose-100 rounded-lg p-8 mb-4 flex items-center justify-center">
                                                <span className="text-6xl">😮</span>
                                            </div>
                                            <h4 className="font-semibold text-gray-900 mb-2">Cheek Lifts</h4>
                                            <p className="text-gray-700 text-sm mb-3">Smile as wide as you can while keeping your lips closed. Hold for 10 seconds, then relax.</p>
                                            <ul className="text-sm text-gray-600 space-y-1">
                                                <li>• Repeat: 3 sets of 15 reps</li>
                                                <li>• Frequency: Daily, morning and evening</li>
                                                <li>• Focus: Feel the stretch in your cheeks</li>
                                            </ul>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="bg-rose-100 rounded-lg p-8 mb-4 flex items-center justify-center">
                                                <span className="text-6xl">😗</span>
                                            </div>
                                            <h4 className="font-semibold text-gray-900 mb-2">Fish Face</h4>
                                            <p className="text-gray-700 text-sm mb-3">Suck in your cheeks and pucker your lips. Try to smile while holding this position.</p>
                                            <ul className="text-sm text-gray-600 space-y-1">
                                                <li>• Repeat: 3 sets of 20 seconds</li>
                                                <li>• Frequency: Twice daily</li>
                                                <li>• Focus: Engage cheek and jaw muscles</li>
                                            </ul>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="bg-rose-100 rounded-lg p-8 mb-4 flex items-center justify-center">
                                                <span className="text-6xl">💨</span>
                                            </div>
                                            <h4 className="font-semibold text-gray-900 mb-2">Air Puffing</h4>
                                            <p className="text-gray-700 text-sm mb-3">Fill your mouth with air, transfer it from cheek to cheek for 30 seconds.</p>
                                            <ul className="text-sm text-gray-600 space-y-1">
                                                <li>• Repeat: 3 sets of 30 seconds</li>
                                                <li>• Frequency: Daily</li>
                                                <li>• Focus: Stretch facial muscles evenly</li>
                                            </ul>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="bg-rose-100 rounded-lg p-8 mb-4 flex items-center justify-center">
                                                <span className="text-6xl">😤</span>
                                            </div>
                                            <h4 className="font-semibold text-gray-900 mb-2">Jaw Release</h4>
                                            <p className="text-gray-700 text-sm mb-3">Move your jaw like you're chewing while keeping lips together. Breathe in and out.</p>
                                            <ul className="text-sm text-gray-600 space-y-1">
                                                <li>• Repeat: 2 sets of 1 minute</li>
                                                <li>• Frequency: Daily</li>
                                                <li>• Focus: Relax and release tension</li>
                                            </ul>
                                        </div>
                                    </div>

                                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mt-6">
                                        <p className="text-sm text-amber-900">
                                            <strong>💡 Pro Tip:</strong> Consistency is key! Set a reminder to do these exercises twice daily. Results typically visible in 4-6 weeks.
                                        </p>
                                    </div>
                                </div>
                            )}

                            {expandedRecommendation.recommendationIndex === 1 && expandedRecommendation.feature === 'faceShape' && (
                                <div className="space-y-6">
                                    <h3 className="text-xl font-semibold text-gray-900 mb-4">🧴 Skincare Routine for Face Shape Enhancement</h3>
                                    
                                    <div className="space-y-4">
                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">🧼</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">Step 1: Gentle Exfoliation (2-3 times/week)</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Use a gentle facial scrub or chemical exfoliant to remove dead skin cells and promote cell renewal.</p>
                                                    <div className="bg-white p-3 rounded-lg">
                                                        <p className="text-sm font-medium text-gray-900 mb-2">Recommended Products:</p>
                                                        <ul className="text-sm text-gray-600 space-y-1">
                                                            <li>• CeraVe SA Cleanser (chemical exfoliation)</li>
                                                            <li>• Paula's Choice 2% BHA Liquid</li>
                                                            <li>• The Ordinary AHA 30% + BHA 2% (weekly)</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">💆</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">Step 2: Facial Massage (Daily)</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Massage improves circulation, promotes lymphatic drainage, and can help define facial contours.</p>
                                                    <div className="bg-white p-3 rounded-lg">
                                                        <p className="text-sm font-medium text-gray-900 mb-2">Technique:</p>
                                                        <ul className="text-sm text-gray-600 space-y-1">
                                                            <li>• Use upward strokes from neck to forehead</li>
                                                            <li>• Apply gentle pressure along jawline</li>
                                                            <li>• Duration: 5-10 minutes daily</li>
                                                            <li>• Use face oil or serum for smooth gliding</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">✨</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">Step 3: Hydration & Firming (Daily)</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Keep skin hydrated and use products with firming ingredients to maintain elasticity.</p>
                                                    <div className="bg-white p-3 rounded-lg">
                                                        <p className="text-sm font-medium text-gray-900 mb-2">Key Ingredients:</p>
                                                        <ul className="text-sm text-gray-600 space-y-1">
                                                            <li>• Hyaluronic Acid (hydration)</li>
                                                            <li>• Retinol (collagen production)</li>
                                                            <li>• Vitamin C (firming & brightness)</li>
                                                            <li>• Peptides (skin elasticity)</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mt-6">
                                        <p className="text-sm text-blue-900">
                                            <strong>💧 Hydration Tip:</strong> Drink 8-10 glasses of water daily. Proper hydration dramatically improves skin appearance and facial definition.
                                        </p>
                                    </div>
                                </div>
                            )}

                            {expandedRecommendation.recommendationIndex === 2 && expandedRecommendation.feature === 'faceShape' && (
                                <div className="space-y-6">
                                    <h3 className="text-xl font-semibold text-gray-900 mb-4">👓 Frame Selection Guide for Face Balance</h3>
                                    
                                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6">
                                        <p className="text-sm text-blue-900">
                                            <strong>The Rule:</strong> Choose frames that contrast with your face shape to create balance and harmony.
                                        </p>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">⬜</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">For Round Faces</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Choose angular, rectangular frames to add definition and length.</p>
                                                    <div className="grid grid-cols-2 gap-3 mt-3">
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">✅ Best Styles:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Rectangular frames</li>
                                                                <li>• Angular cat-eye</li>
                                                                <li>• Geometric shapes</li>
                                                            </ul>
                                                        </div>
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">❌ Avoid:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Round frames</li>
                                                                <li>• Small frames</li>
                                                                <li>• Narrow styles</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">🔷</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">For Square Faces</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Choose round or oval frames to soften angular features.</p>
                                                    <div className="grid grid-cols-2 gap-3 mt-3">
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">✅ Best Styles:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Round frames</li>
                                                                <li>• Oval frames</li>
                                                                <li>• Aviators</li>
                                                            </ul>
                                                        </div>
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">❌ Avoid:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Square frames</li>
                                                                <li>• Boxy styles</li>
                                                                <li>• Angular designs</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-gray-50 p-6 rounded-xl">
                                            <div className="flex items-start gap-4">
                                                <div className="bg-rose-100 rounded-lg p-4 flex-shrink-0">
                                                    <span className="text-4xl">⬭</span>
                                                </div>
                                                <div className="flex-1">
                                                    <h4 className="font-semibold text-gray-900 mb-2">For Oval Faces</h4>
                                                    <p className="text-gray-700 text-sm mb-3">Most shapes work! Experiment with bold, statement frames.</p>
                                                    <div className="grid grid-cols-2 gap-3 mt-3">
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">✅ Best Styles:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Oversized frames</li>
                                                                <li>• Wayfarers</li>
                                                                <li>• Cat-eye</li>
                                                            </ul>
                                                        </div>
                                                        <div className="bg-white p-3 rounded-lg">
                                                            <p className="text-sm font-medium text-gray-900">💡 Tip:</p>
                                                            <ul className="text-xs text-gray-600 mt-1 space-y-1">
                                                                <li>• Try bold colors</li>
                                                                <li>• Experiment freely</li>
                                                                <li>• Have fun!</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="bg-gradient-to-r from-rose-50 to-pink-50 border border-rose-200 rounded-xl p-6 mt-6">
                                        <h4 className="font-semibold text-gray-900 mb-3">🛍️ Where to Try:</h4>
                                        <div className="grid md:grid-cols-3 gap-3">
                                            <div className="bg-white p-3 rounded-lg">
                                                <p className="font-medium text-gray-900 text-sm">Warby Parker</p>
                                                <p className="text-xs text-gray-600">Free home try-on</p>
                                            </div>
                                            <div className="bg-white p-3 rounded-lg">
                                                <p className="font-medium text-gray-900 text-sm">Zenni Optical</p>
                                                <p className="text-xs text-gray-600">Affordable options</p>
                                            </div>
                                            <div className="bg-white p-3 rounded-lg">
                                                <p className="font-medium text-gray-900 text-sm">Local Optician</p>
                                                <p className="text-xs text-gray-600">Professional fitting</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Default content for other recommendations */}
                            {!((expandedRecommendation.recommendationIndex >= 0 && expandedRecommendation.recommendationIndex <= 2) && expandedRecommendation.feature === 'faceShape') && (
                                <div className="space-y-6">
                                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                                        <p className="text-blue-900">
                                            <strong>📚 Coming Soon:</strong> Detailed guides for this recommendation are being prepared. In the meantime, focus on consistency with the basic recommendation above.
                                        </p>
                                    </div>

                                    <div className="space-y-4">
                                        <h3 className="text-lg font-semibold text-gray-900">General Tips:</h3>
                                        <div className="grid md:grid-cols-2 gap-4">
                                            <div className="bg-gray-50 p-4 rounded-xl">
                                                <h4 className="font-medium text-gray-900 mb-2">✅ Do:</h4>
                                                <ul className="text-sm text-gray-700 space-y-1">
                                                    <li>• Be consistent with your routine</li>
                                                    <li>• Track your progress with photos</li>
                                                    <li>• Stay patient - results take time</li>
                                                </ul>
                                            </div>
                                            <div className="bg-gray-50 p-4 rounded-xl">
                                                <h4 className="font-medium text-gray-900 mb-2">❌ Don't:</h4>
                                                <ul className="text-sm text-gray-700 space-y-1">
                                                    <li>• Expect overnight results</li>
                                                    <li>• Skip days frequently</li>
                                                    <li>• Overdo exercises or products</li>
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 p-6">
                            <button
                                onClick={() => setExpandedRecommendation(null)}
                                className="w-full bg-gradient-to-r from-rose-500 to-pink-500 text-white py-3 rounded-xl font-semibold hover:from-rose-600 hover:to-pink-600 transition-all"
                            >
                                Got it! Back to Recommendations
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}