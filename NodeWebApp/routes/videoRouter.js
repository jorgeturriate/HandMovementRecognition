const express= require('express');
const router= express.Router();
const {v4:uuidv4} = require('uuid');


//Handling the get request to root
router.get('/home', (req, res)=>{
    res.render('homeindex')
});


router.get('/hands/:room', (req, res)=>{
    res.render('index', {RoomId:req.params.room})
})


router.get('/hands', (req, res)=>{
    res.render('handindex')
})


module.exports= router;