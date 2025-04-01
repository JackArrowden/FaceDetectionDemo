import { useNavigate } from 'react-router'
import bg_img from '../../assets/bg/2.jpg'
import './home.css'

function Home() {
    const nav = useNavigate()

    return (
        <div className="h-full w-full overflow-hidden flex relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg"/>

            <div className="box-container">
                <div className="box-border"/>
                <div className="box-content">
                    <p 
                        className="flex justify-center items-center text-6xl w-2/5 h-4/5 text-[#F49393] z-20 cursor-pointer hover:opacity-60"
                        onClick={() => nav('/demo')}
                    >
                        Demo
                    </p>
                    <div className="w-[2px] h-4/5 bg-[#FFE6E6] rounded-full mx-[5%]"/>
                    <p 
                        className="flex justify-center items-center text-6xl w-2/5 h-4/5 text-[#F49393] z-20 cursor-pointer hover:opacity-60"
                        onClick={() => nav('/about')}
                    >
                        About
                    </p>
                </div>
                <div className="box-border"/>
            </div>
        </div>
    )
}

export default Home