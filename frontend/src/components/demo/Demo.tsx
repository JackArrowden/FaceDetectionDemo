// Main libs
import React, { useState, useRef, useEffect } from 'react'
import { ArrowDown, ArrowUp, RefreshCcw, RefreshCw } from 'lucide-react'
import { HashLoader } from 'react-spinners'

// Others
import bg_img from '../../assets/bg/2.jpg'
import BackBtn from '../about/BackBtn'
import './demo.css'

function DemoPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [isApiComplete, setIsApiComplete] = useState<boolean>(false)
    const [originalImage, setOriginalImage] = useState<string | null>(null)
    const [resultData, setResultData] = useState<string | null>(null)
    const [resultType, setResultType] = useState<'image' | 'video' | null>(null)
    const [faceCount, setFaceCount] = useState<number>(0)
    const [showOriginal, setShowOriginal] = useState<boolean>(false)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleAddFileClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click()
        }
    }

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (file) {
            setSelectedFile(file)
            setIsApiComplete(false)
            setResultData(null)
            setResultType(null)
            setFaceCount(0)
            setShowOriginal(false)

            const reader = new FileReader()
            reader.onload = () => {
                setOriginalImage(reader.result as string)
            }
            reader.readAsDataURL(file)

            await sendFileToApi(file)
        }
    }

    const sendFileToApi = async (file: File) => {
        const formData = new FormData()
        formData.append('file', file)

        try {
            const response = await fetch('http://localhost:8000/detect', {
                method: 'POST',
                body: formData,
            })

            if (response.ok) {
                const result = await response.json()
                const { type, data, num_faces } = result

                const dataUrl = type === 'image' ? `data:image/jpeg;base64,${data}` : `data:video/mp4;base64,${data}`

                setResultData(dataUrl)
                setResultType(type)
                setFaceCount(Math.round(num_faces))
                setIsApiComplete(true)
            } else {
                console.error('API error:', response.statusText)
            }
        } catch (error) {
            console.error('Error sending file to API:', error)
        }
    }

    const handleDownload = () => {
        if (resultData) {
            const link = document.createElement('a')
            link.href = resultData
            link.download = resultType === 'image' ? 'processed_image.jpg' : 'processed_video.mp4'
            link.click()
        }
    }
    
    useEffect(() => {
        return () => {
            if (resultData) URL.revokeObjectURL(resultData)
            if (originalImage) URL.revokeObjectURL(originalImage)
        }
    }, [resultData, originalImage])

    return (
        <div className="h-full w-full overflow-hidden flex flex-col relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg" />
            <BackBtn />

            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*,video/*"
                className="hidden"
                aria-label='fileInput'
            />

            {/* 
                =============================== FILE INPUT ===============================
            */}
            <div className="flex items-center gap-4 z-10">
                {selectedFile && (
                    <div className="flex justify-start items-center bg-[rgba(237,256,236,0.1)] w-[30dvw] h-12 rounded-xl border-1 border-[#51F83B]">
                        <p className="text-white m-4 font-semibold text-lg">{selectedFile.name}</p>
                    </div>
                )}
                <div
                    onClick={() => handleAddFileClick()}
                    className="flex justify-center items-center gap-2 px-6 py-3 bg-[#51F83B] hover:bg-[#6cae63] rounded-full transition-all duration-300 cursor-pointer"
                >
                    <p className="text-black font-semibold text-xl">Add File</p>
                    <ArrowUp color='black' size={28} />
                </div>
            </div>

            {/* 
                =============================== PROGRESS BAR ===============================
            */}
            {selectedFile && !isApiComplete && (
                <div className='flex flex-col items-center'>
                    <HashLoader
                        size={75}
                        color='#2ae8b4'
                        className="mt-16 mb-8"
                    />
                    <p className="text-white font-semibold text-xl z-10">Hold on. Your file is being processed...</p>
                </div>
            )}

            {/* 
                =============================== RESULT DATA ===============================
            */}
            {resultData && (
                <div className="my-8 z-10">
                    <p className="text-white font-semibold text-3xl">
                        The total number of faces is: {faceCount}
                    </p>
                    
                    <div className="flex justify-center items-center mt-8">
                        {resultType == 'image' &&
                            <div 
                                onClick={() => setShowOriginal(!showOriginal)}
                                className="flex bg-[#51F83B] justify-center items-center rounded-full w-12 h-12 cursor-pointer hover:bg-[#6cae63]"
                            >
                                {showOriginal ? (
                                    <RefreshCw color='black' size={28} />
                                ) : (
                                    <RefreshCcw color='black' size={28} />
                                )}
                            </div>
                        }

                        <p className="text-white text-2xl font-semibold mx-4 w-64 text-center">{showOriginal ? 'Before YOLO-FaceV2' : 'After YOLO-FaceV2'}</p>
                        
                        <abbr title='Download result file'>
                            <div 
                                onClick={() => handleDownload()}
                                className="flex bg-[#51F83B] justify-center items-center rounded-full w-12 h-12 cursor-pointer hover:bg-[#6cae63]"
                            >
                                <ArrowDown color='black' size={28} />
                            </div>
                        </abbr>
                    </div>
                </div>
            )}

            {resultData && (showOriginal ? (
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
                    className="max-w-144 max-h-72 object-contain rounded-lg z-10"
                />
            ))}
        </div>
    )
}

export default DemoPage