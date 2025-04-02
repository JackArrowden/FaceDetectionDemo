import bg_img from '../../assets/bg/2.jpg'
import './about.css'
import BackBtn from './BackBtn'

function AboutPage() {
    return (
        <div className="h-full w-full overflow-hidden flex relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg"/>
            <BackBtn />
        </div>
    )
}

export default AboutPage