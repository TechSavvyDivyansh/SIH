import express from 'express'
import chatCotroller from '../controller/chat_controller.js'

const router=express.Router()

router.get('/',chatCotroller.saveChat)

export default router