import bg_img from '../../assets/bg/2.jpg'
import BackBtn from '../about/BackBtn'
import './demo.css'

function DemoPage() {
    return (
        <div className="h-full w-full overflow-hidden flex relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg"/>
            <BackBtn />

            {/* Your code goes here */}
        </div>
    )
}

export default DemoPage