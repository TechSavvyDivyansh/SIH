import express from 'express'
import mongoose from 'mongoose'
import dotenv from 'dotenv'
import cors from 'cors'
import ChatRouter from '../api/routes/chat_route.js'


dotenv.config()

mongoose.connect(process.env.MONGO).then(()=>{
    console.log("mongodb connected")
}).catch((err)=>{
    console.log(err)
})


const app=express()

app.use(express.json())
app.use(cors());

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}!!`)
})

app.use('/api',ChatRouter)