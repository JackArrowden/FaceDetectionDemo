import { ChevronLeft } from "lucide-react"
import { useNavigate } from "react-router"
import './backBtn.css'

function BackBtn() {
    const nav = useNavigate()

    return (
        <div 
            className="backbtn flex flex-row z-20 absolute top-[6dvh] left-[10dvw] justify-center items-center cursor-pointer hover:opacity-80"
            onClick={() => nav('/')}
        >
            <ChevronLeft size={36} className="mr-4" color="white"/>
            <p className="text-4xl text-white">Back</p>
        </div>
    )
}

export default BackBtn