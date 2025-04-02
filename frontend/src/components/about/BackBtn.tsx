import { ChevronLeft } from "lucide-react"
import { useNavigate } from "react-router"

function BackBtn() {
    const nav = useNavigate()

    return (
        <div 
            className="flex flex-row z-10 absolute top-[6dvh] left-[10dvw] justify-center items-center cursor-pointer hover:opacity-80"
            onClick={() => nav('/')}
        >
            <ChevronLeft size={36} className="mr-4"/>
            <p className="text-4xl text-white">Back</p>
        </div>
    )
}

export default BackBtn