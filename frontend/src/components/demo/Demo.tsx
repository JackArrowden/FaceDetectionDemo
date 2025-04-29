import { ArrowDown, ArrowUp, RotateCcw, RotateCw } from 'lucide-react';
import bg_img from '../../assets/bg/2.jpg';
import BackBtn from '../about/BackBtn';
import './demo.css';
import { useState, useRef, useEffect } from 'react';

function DemoPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [progress, setProgress] = useState<number>(0);
    const [isApiComplete, setIsApiComplete] = useState<boolean>(false);
    const [originalImage, setOriginalImage] = useState<string | null>(null);
    const [resultData, setResultData] = useState<string | null>(null);
    const [resultType, setResultType] = useState<'image' | 'video' | null>(null);
    const [faceCount, setFaceCount] = useState<number>(0);
    const [showOriginal, setShowOriginal] = useState<boolean>(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleAddFileClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file); // Lưu file vào state
            setProgress(0); // Reset progress
            setIsApiComplete(false); // Reset trạng thái API
            setResultData(null); // Reset dữ liệu kết quả
            setResultType(null); // Reset loại file
            setFaceCount(0); // Reset số khuôn mặt
            setShowOriginal(false); // Reset trạng thái hiển thị

            // Lưu URL của ảnh gốc
            const originalUrl = URL.createObjectURL(file);
            setOriginalImage(originalUrl);

            // Gửi file đến API
            await sendFileToApi(file);
        }
    };

    const sendFileToApi = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/detect', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                const { type, data, num_faces } = result;

                const dataUrl = type === 'image' ? `data:image/jpeg;base64,${data}` : `data:video/mp4;base64,${data}`;

                setResultData(dataUrl);
                setResultType(type);
                setFaceCount(Math.round(num_faces));
                setIsApiComplete(true);
                setProgress(100);
            } else {
                console.error('API error:', response.statusText);
                setProgress(0);
            }
        } catch (error) {
            console.error('Error sending file to API:', error);
            setProgress(0);
        }
    };

    const handleReverse = () => {
        setShowOriginal((prev) => !prev);
    };

    const handleDownload = () => {
        if (resultData) {
            const link = document.createElement('a');
            link.href = resultData;
            link.download = resultType === 'image' ? 'processed_image.jpg' : 'processed_video.mp4';
            link.click();
        }
    };

    useEffect(() => {
        if (selectedFile && !isApiComplete) {
            const interval = setInterval(() => {
                setProgress(prevProgress => {
                    const min = prevProgress + 1;
                    const max = 99;
                    if (min > max) return prevProgress; // Đã gần 100 rồi thì giữ nguyên
                    const randomProgress = Math.floor(Math.random() * (max - min + 1)) + min;
                    return randomProgress;
                });
            }, 500);
    
            return () => clearInterval(interval);
        }
    }, [selectedFile, isApiComplete]);
    
    useEffect(() => {
        return () => {
            if (resultData) {
                URL.revokeObjectURL(resultData);
            }
            if (originalImage) {
                URL.revokeObjectURL(originalImage);
            }
        };
    }, [resultData, originalImage]);

    return (
        <div className="h-full w-full overflow-hidden flex flex-col relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg" />
            <BackBtn />

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*,video/*" // Chỉ cho phép ảnh và video
                className="hidden"
                aria-label='fileInput'
            />

            <div className="flex items-center gap-4 z-10">
                {selectedFile && (
                    <div className="flex justify-start items-center bg-[rgba(237,256,236,0.1)] w-[30dvw] h-12 rounded-xl border-1 border-[#51F83B]">
                        <p className="text-white m-4 font-semibold text-lg">{selectedFile.name}</p>
                    </div>
                )}
                <div
                    onClick={handleAddFileClick}
                    className="flex justify-center items-center gap-2 px-6 py-3 bg-[#51F83B] hover:bg-[#6cae63] rounded-full transition-all duration-300 cursor-pointer"
                >
                    <p className="text-black font-semibold text-xl">Add File</p>
                    <ArrowUp color='black' size={28} />
                </div>
            </div>

            {selectedFile && !isApiComplete && (
                <div className="flex mt-6 justify-center items-center gap-4 w-100 h-6 rounded-full z-10">
                    <div className="relative w-[80%] h-full bg-[rgba(237,256,236,0.1)] border-1 border-[#51F83B] rounded-full overflow-hidden">
                        <div
                            className="absolute top-0 left-0 h-full bg-[#51F83B]"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                    <p className="text-white rounded-full bg-[rgba(196,255,176,0.15)] text-center w-20 font-semibold text-xl">{progress}%</p>
                </div>
            )}

            {resultData && (
                <div className="my-8 z-10">
                    <p className="text-white font-semibold text-3xl">
                        The number of face is: {faceCount}
                    </p>
                    <div className="flex justify-center items-center mt-8">
                        {
                        resultType == 'image' &&
                        (showOriginal ? (
                            <div className="flex bg-[#51F83B] justify-center items-center rounded-full w-12 h-12 cursor-pointer hover:bg-[#6cae63]">
                                <RotateCcw size={28} onClick={handleReverse} />
                            </div>
                        ) : (
                            <div className="flex bg-[#51F83B] justify-center items-center rounded-full w-12 h-12 cursor-pointer hover:bg-[#6cae63]">
                                <RotateCw size={28} onClick={handleReverse} />
                            </div>
                        ))}
                        <p className="text-white text-2xl font-semibold mx-4">After YOLO-FaceV2</p>
                        <div className="flex bg-[#51F83B] justify-center items-center rounded-full w-12 h-12 cursor-pointer hover:bg-[#6cae63]">
                            <ArrowDown size={28} onClick={handleDownload} />
                        </div>
                    </div>
                </div>
            )}

            { resultData && (showOriginal ? (
                    <img
                        src={originalImage!}
                        alt="Original Image"
                        className="max-w-144 max-h-72 object-contain rounded-lg z-10"
                    />
                ) : resultType === 'image' ? (
                    <img
                        src={resultData!}
                        alt="Processed Image"
                        className="max-w-144 max-h-72 object-contain rounded-lg z-10"
                    />
                ) :  (
                    <video
                        src={resultData!}
                        controls
                        // autoPlay
                        className="max-w-144 max-h-72 object-contain rounded-lg z-10"
                    />
                )) 
            }
        </div>
    )
}

export default DemoPage