// Main libs
import BackBtn from './BackBtn'
import { teamInfo } from '../../assets/data/teamInfo'
import './about.css'

// Others
import bg_img from '../../assets/bg/2.jpg'
import { UserRound } from 'lucide-react'

function AboutPage() {
    return (
        <div className="page h-full w-full overflow-hidden flex flex-col relative justify-center items-center">
            <img src={bg_img} alt="" className="web-bg"/>
            <BackBtn />

            <div className="page-content">
                <div className="about-header rounded-xl my-2">
                    <p className="media uppercase text-4xl my-2 mx-16 text-white">about this demo</p>
                </div>

                <div className="about-content-cont rounded-xl w-9/12 my-1 p-6">
                    <p className="about-content-header text-4xl mb-2 text-white">What is YOLO FaceV2?</p>
                    <p className="about-content-content text-xl mt-2 text-white">
                        As a variant of YOLOv5, YOLO Face V2 was designed to enhance face detection performance under challenging conditions such as varying scales, angles, or partial occlusions. As a modern and optimized version in the field of facial recognition, YOLO Face V2 was selected by our team to be featured in this demo. With its user-friendly and intuitive interface, the system allows users to obtain face detection results directly from the uploaded original image. This demo not only demonstrates the model’s effectiveness in real-world scenarios but also helps users better understand the strengths and limitations of YOLO Face V2 across different conditions.
                    </p>
                </div>

                <div className="about-content-cont2 flex flex-col justify-center rounded-xl w-9/12 my-1 px-6">
                    <p className="about-content-header text-4xl mb-2 text-white">Our team</p>
                    <div className="mt-2 flex flex-row justify-center flex-wrap">
                        {teamInfo.map((member, index) => (
                            <div 
                                key={index}
                                className="about-item"
                                onClick={() => window.open(member.github, "_blank")}
                            >
                                <div className="about-subitem flex justify-center items-center w-14 h-14 rounded-full bg-[#51F83B] mr-4">
                                    <UserRound size={32} color='black'/>  
                                </div>
                                <div key={index} className="flex flex-col">
                                    <p className="text-xl text-white">{member.name}</p>
                                    <p className="text-xl text-white">{member.id}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default AboutPage